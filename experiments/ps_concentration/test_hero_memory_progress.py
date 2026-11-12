from .hero_memory_progress import prior_notes, summarize_opportunities, valid_choices_by_turn


class Characters:
    def decode(self, tokens, **kwargs):
        return ''.join(map(chr,tokens))


def test_prior_notes_use_exact_previous_window_and_exclude_current_thinking_prefix_and_actions():
    text='aA=7\n'+'aA=7\n'+'z'*20+'aB=3\nAction: aC=4\n'+'aD=5\n'
    start=len(text)-len('aD=5\n')
    record=dict(tokens=list(map(ord,text)),roles=['assistant']*len(text),prefix_length=5,window=18)
    notes=prior_notes(record,Characters(),start)
    assert [(n['position'],n['rank']) for n in notes]==[('aB','3')]
    assert all(start-18 <= n['start'] < n['end'] <= start for n in notes)
    record['window']=len(text)
    notes=prior_notes(record,Characters(),start)
    assert sum(n['position']=='aA' for n in notes)==1
    assert not any(n['position'] in ('aC','aD') for n in notes)


def test_opportunity_counts_include_failures_use_strict_age_boundaries_and_same_state_chance():
    rows=[dict(age_before_thinking=age,prior_note=note,correct=correct,valid_choices=choices)
        for age,note,correct,choices in [(1024,True,True,15),(1025,True,True,13),
            (2048,False,False,3),(2049,False,True,1),(2050,False,False,5)]]
    groups=summarize_opportunities(rows,1024)
    assert groups['>W']['all']['all_states']['opportunities']==4
    assert groups['>W']['no_recognized_prior_note']['all_states']['successes']==1
    assert groups['>W']['no_recognized_prior_note']['all_states']['opportunities']==3
    assert groups['>2W']['all']['all_states']['opportunities']==2
    assert groups['(W,2W]']['all']['all_states']['opportunities']==2
    nonforced=groups['>W']['no_recognized_prior_note']['nonforced_states']
    assert nonforced['opportunities']==2 and nonforced['successes']==0
    assert nonforced['random_valid_rate']==(1/3+1/5)/2


def test_random_valid_choices_follow_native_matches_and_preserve_invalid_move_opportunities():
    turns=[]
    for rank in range(8):
        for second in (False,True):
            turns.append(dict(round=rank,parse=dict(coord=f'{rank}-{int(second)}'),
                env_result=dict(applied=True,revealed_face=rank,phase_after='flip_first' if second else 'flip_second')))
    turns.insert(3,dict(round=1,parse=dict(coord='invalid'),env_result=dict(applied=False)))
    choices=valid_choices_by_turn(dict(config=dict(rows=4,cols=4,num_cards=8),turns=turns))
    assert choices[1]==15
    assert choices[3]==choices[4]==13
    assert choices[len(turns)-1]==1
