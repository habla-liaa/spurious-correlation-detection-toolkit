import re

from collections import defaultdict


def read_config_from_path(path, base_path):
    experiment_config = path.replace(base_path, '').split('/')
    parsed = {}
    for text in experiment_config:
        if not text or '.pkl' in text:
            continue
        parts = text.split('-', 1)
        config_key = parts[0]
        config_value = parts[1] if len(parts) > 1 else ''
        parsed[config_key] = config_value
    return parsed


def get_common_tokens_with_root(process_names):
    groups = defaultdict(list)
    for name in process_names:
        tokens = re.split('-', name)
        if not tokens:
            continue
        process_type = tokens[0]
        
        groups[process_type].append({
            'tokens': tokens[1:],
            'root': process_type
        })
    
    root_common_tokens = {}
    for root, entries in groups.items():
        if not entries:
            continue
        common_tokens = set(entries[0]['tokens'])
        for entry in entries[1:]:
            common_tokens.intersection_update(entry['tokens'])
        root_common_tokens[root] = common_tokens
    
    return root_common_tokens


def compute_short_name_with_root(process_name, root_common_tokens):
    if not process_name:
        return process_name or ''

    tokens = re.split('-', process_name)
    process_type = tokens[0]
    if process_type not in root_common_tokens:
        return process_name 
    common_tokens = root_common_tokens[process_type]
    short_tokens = [token for token in tokens if token not in common_tokens]
    short_name = '-'.join(short_tokens) if short_tokens else process_type
    return short_name


def get_common_tokens(process_names):
    clean_tokens = [re.split('-', name) for name in process_names]
    if not clean_tokens:
        return set()

    common_tokens = set(clean_tokens[0])
    for tokens in clean_tokens[1:]:
        common_tokens.intersection_update(tokens)
    return common_tokens


def compute_short_name(process_name, common_tokens):
    if not process_name:
        return ''

    tokens = re.split('-', process_name)
    short_tokens = [token for token in tokens if token not in common_tokens]
    short_name = '-'.join(short_tokens) if short_tokens else ''
    return short_name
