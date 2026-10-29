import os
import json
from pathlib import Path
import time


class Handoff(Exception):
    pass


def handoff_requested():
    path=os.environ.get('PS_CONCENTRATION_HANDOFF')
    allocation=os.environ.get('ALLOCATION_ROOT')
    shared=False
    if path and Path(path).exists():
        content=Path(path).read_text()
        if content:
            request=json.loads(content)
            owner=str(request.get('owner_job'))
            shared=owner==str(os.environ.get('SLURM_JOB_ID')) or (
                owner=='*' and str(request.get('requester_job'))!=str(os.environ.get('SLURM_JOB_ID')))
    deadline=os.environ.get('PS_CONCENTRATION_END_EPOCH')
    margin=float(os.environ.get('PS_CONCENTRATION_HANDOFF_MARGIN',1800))
    expiring=bool(deadline and time.time()>=float(deadline)-margin)
    return bool(shared or expiring or (allocation and (Path(allocation)/'shutdown.requested').exists()))


def check_handoff():
    if handoff_requested():
        raise Handoff('Allocation handoff requested; all committed state is preserved')
