def tweak_query(query : str, model : str):
    """
    for qwen3 models, turn off thinking
    """
    
    # Check if the model is qwen3
    if 'qwen3' in model:
        # Check if the query contains '/no_think'
        if '/no_think' not in query:
            # Append '/no_think' to the query
            query += '\n/no_think'
    return query