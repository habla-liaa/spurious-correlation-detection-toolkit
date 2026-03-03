def get_name(params):
    name = 'clip'
    for c, v in params.items():
        name += f'-{c[0]}_{v}'
    return name


def clip_alignment(aligns, params):    
    aligns['duration'] = aligns['end'] - aligns['start']
    sample_duration = aligns.groupby('sample_id').duration.sum().reset_index()
    if params['type'] == 'mean':
        threshold = sample_duration['duration'].mean()
    elif params['type'] == 'median':
        threshold = sample_duration['duration'].median()
    elif params['type'] == 'quantile':
        threshold = sample_duration['duration'].quantile(params['quantile'])
    elif params['type'] == 'fixed':
        threshold = params['value']
    else:
        raise ValueError(f"Unknown clip type: {params['type']}")
    threshold = round(threshold, 2)
        
    sample_to_keep = sample_duration[sample_duration['duration'] >= threshold]['sample_id'].tolist()
    clipped_aligns = aligns[aligns.sample_id.isin(sample_to_keep)].copy()
    clipped_aligns = clipped_aligns.reset_index(drop=True)

    if params['keep_from'] == 'start':
        to_discard = []
        for _, group in clipped_aligns.groupby('sample_id'):
            acum = 0
            group = group.sort_values('start', ascending=params['keep_from'] == 'start')
            for idx, row in group.iterrows():
                if acum >= threshold:
                    to_discard.append(idx)
                elif acum < threshold and acum + row['duration'] >= threshold:
                    clipped_aligns.at[idx, 'end'] = threshold - acum + row['start']
                    acum += row['duration']
                else:
                    acum += row['duration']
        clipped_aligns = clipped_aligns.drop(to_discard)
    else:
        raise ValueError(f"Unknown keep_from method: {params['keep_from']}")
    
    clipped_aligns['duration'] = clipped_aligns['end'] - clipped_aligns['start']
    assert all(clipped_aligns.groupby('sample_id').duration.sum().round(2) == threshold), "WARNING:: Clipping did not result in the expected duration for all samples. Please check the parameters and the resulting alignments."
    
    return clipped_aligns
