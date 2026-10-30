"""Account Slurm GPU allocations, preserving requeues and cleanup without summing steps."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import subprocess


FIELDS = ['JobIDRaw', 'JobName', 'State', 'Start', 'End', 'ElapsedRaw', 'AllocTRES',
          'ExitCode', 'Restarts', 'Partition', 'NodeList']


def timestamp(value):
    if value in ('', 'None', 'Unknown', 'N/A'):
        return None
    result = datetime.fromisoformat(value)
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def gpu_resources(tres):
    entries = dict(item.split('=', 1) for item in tres.split(',') if '=' in item)
    typed = {key.removeprefix('gres/gpu:'): int(value)
             for key, value in entries.items() if key.startswith('gres/gpu:')}
    count = int(entries['gres/gpu']) if 'gres/gpu' in entries else sum(typed.values())
    if typed and sum(typed.values()) != count:
        raise ValueError('Slurm generic and typed GPU counts disagree')
    return count, typed or ({'unknown':count} if count else {})


def interval(row):
    start = timestamp(row['Start'])
    seconds = int(row['ElapsedRaw'] or 0)
    if start is None:
        return None
    end = timestamp(row['End'])
    if seconds == 0 and (end is None or end <= start):
        return None
    return start, max(end or start, start + timedelta(seconds=seconds))


def account(text, *, observed_utc, reservations=None):
    rows = list(csv.DictReader(io.StringIO(text), fieldnames=FIELDS, delimiter='|'))
    roots = [row for row in rows if row['JobIDRaw'].isdigit()]
    steps = [row for row in rows if '.' in row['JobIDRaw']]
    intervals, issues, restart_records, states = {}, [], {}, {}
    for row in roots:
        job = row['JobIDRaw']
        restart_records.setdefault(job, set()).add(int(row['Restarts'] or 0))
        states[job] = row['State']
        span = interval(row)
        gpus, kinds = gpu_resources(row['AllocTRES'])
        if span is None or not gpus:
            continue
        start, end = span
        key = job, start.isoformat()
        if key in intervals:
            previous = intervals[key]
            if previous['gpus'] != gpus or previous['gpu_types'] != kinds:
                raise ValueError('The same allocation start has conflicting GPU resources')
            previous['end'] = max(previous['end'], end)
            previous['states'].add(row['State'])
            previous['root_elapsed_seconds'] = max(previous['root_elapsed_seconds'], int(row['ElapsedRaw']))
            continue
        intervals[key] = dict(job_id=job, job_name=row['JobName'], start=start, end=end,
            partition=row['Partition'], nodes=row['NodeList'],
            parent_end=end, gpus=gpus, gpu_types=kinds, states={row['State']},
            root_elapsed_seconds=int(row['ElapsedRaw']), matched_steps=0)
    for row in steps:
        span = interval(row)
        if span is None:
            continue
        start, end = span
        job = row['JobIDRaw'].split('.', 1)[0]
        candidates = [value for value in intervals.values() if value['job_id'] == job]
        overlaps = [max(0., (min(value['end'], end) - max(value['start'], start)).total_seconds()) for value in candidates]
        if not candidates or max(overlaps) == 0:
            issues.append(f'Unassigned Slurm step {row["JobIDRaw"]} at {start.isoformat()}')
            continue
        parent = candidates[overlaps.index(max(overlaps))]
        parent['start'], parent['end'] = min(parent['start'], start), max(parent['end'], end)
        parent['matched_steps'] += 1
    for job, indices in restart_records.items():
        missing = set(range(max(indices) + 1)) - indices
        if missing:
            issues.append(f'Job {job} lacks accounting records for restart indices {sorted(missing)}')
    allocations = []
    by_type = {}
    for value in sorted(intervals.values(), key=lambda item: (item['start'], item['job_id'])):
        seconds = (value['end'] - value['start']).total_seconds()
        allocation = dict(value, start=value['start'].isoformat(), end=value['end'].isoformat(),
            parent_end=value['parent_end'].isoformat(), states=sorted(value['states']),
            allocated_wall_seconds=seconds, cleanup_extension_seconds=(value['end'] - value['parent_end']).total_seconds(),
            gpu_hours=seconds * value['gpus'] / 3600)
        allocations.append(allocation)
        for kind, count in value['gpu_types'].items():
            by_type[kind] = by_type.get(kind, 0.) + seconds * count / 3600
    pending = {}
    for job, reservation in (reservations or {}).items():
        if states.get(str(job)) == 'PENDING':
            pending[str(job)] = dict(reservation, maximum_gpu_hours=reservation['gpus'] * reservation['max_hours'])
    return dict(observed_utc=observed_utc, source_timezone='UTC', allocation_records=allocations,
        total_allocated_gpu_hours=sum(row['gpu_hours'] for row in allocations), gpu_hours_by_type=by_type,
        pending_reservations=pending, pending_maximum_gpu_hours=sum(row['maximum_gpu_hours'] for row in pending.values()),
        issues=issues, complete_accounting_for_requested_records=not issues,
        scope='Actual allocated wall time, including setup, idle, failures, requeues and step cleanup. Pending jobs cost zero. Hardware types are reported separately; GPU-hours do not normalize their speed.',
        step_rule='Batch/extern intervals extend their parent allocation; step durations are never added as separate GPU allocations')


def save(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'allocation-costs.json').write_text(json.dumps(result, indent=2) + '\n')
    fields = ['job_id', 'job_name', 'partition', 'nodes', 'start', 'end', 'states', 'gpus', 'gpu_types',
              'allocated_wall_seconds', 'cleanup_extension_seconds', 'gpu_hours']
    with (output / 'allocation-costs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['allocation_records'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', nargs='+', required=True)
    parser.add_argument('--since', default='2026-10-29T00:00:00')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reservations', help='JSON mapping job ID to {gpus,max_hours}; pending reservations only')
    parser.add_argument('--input', help='Read a previously captured pipe-delimited sacct export instead of querying Slurm')
    args = parser.parse_args()
    if any(not job.isdigit() for job in args.jobs):
        parser.error('Supply explicit numeric allocation IDs belonging to this experiment')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    formats = [f'{name}%180' if name in ('JobName', 'AllocTRES') else name for name in FIELDS]
    command = ['sacct', '--duplicates', '-n', '-P', '-S', args.since, '-j', ','.join(args.jobs),
               '-o', ','.join(formats)]
    raw = Path(args.input).read_text() if args.input else subprocess.check_output(command, text=True,
        env=dict(os.environ, TZ='UTC'))
    (output / 'sacct-records.psv').write_text(raw)
    observed = datetime.now(timezone.utc).isoformat()
    reservations = json.loads(Path(args.reservations).read_text()) if args.reservations else {}
    result = account(raw, observed_utc=observed, reservations=reservations)
    returned = {row['JobIDRaw'] for row in csv.DictReader(io.StringIO(raw), fieldnames=FIELDS, delimiter='|')
                if row['JobIDRaw'].isdigit()}
    missing = set(args.jobs) - returned
    if missing:
        result['issues'].append(f'No accounting rows returned for requested jobs {sorted(missing)}')
        result['complete_accounting_for_requested_records'] = False
    result['query'] = command
    result['requested_job_ids'] = args.jobs
    save(result, output)
    print(json.dumps({key: value for key, value in result.items() if key not in ('allocation_records', 'query')}, indent=2))


if __name__ == '__main__':
    main()
