"""Score recorded native actions without revealing hidden state to the policy."""


def score_prefix(game, max_responses):
    turns=game['turns'][:max_responses]
    first=None
    last_round=None
    matched=set()
    valid=invalid=format_errors=0
    for turn in turns:
        if turn['round']!=last_round:
            first=None
        last_round=turn['round']
        result=turn['env_result']
        if not result['applied']:
            if turn['parse']['ok']:
                invalid+=1
            else:
                format_errors+=1
            continue
        assert turn['parse']['ok']
        valid+=1
        position=turn['parse']['coord']
        rank=result['revealed_face']
        assert position not in matched
        if result['phase_after']=='flip_second':
            assert first is None
            first=(position,rank)
        else:
            assert result['phase_after']=='flip_first' and first is not None
            if first[0]!=position and first[1]==rank:
                matched.update((first[0],position))
            first=None
    total_pairs=game['config']['total_pairs']
    pairs=len(matched)//2
    done=pairs==total_pairs
    terminal=done or len(turns)==max_responses
    assert valid+invalid+format_errors==len(turns)
    return dict(score=pairs,total_pairs=total_pairs,matched_fraction=pairs/total_pairs,done=done,
                response_count=len(turns),rounds_played=last_round or 0,valid_flips=valid,
                invalid_actions=invalid,format_errors=format_errors,max_responses=max_responses,
                terminal=terminal,turn_cap_failure=terminal and not done,
                terminal_outcome_reward=(1. if done else -1.) if terminal else None)
