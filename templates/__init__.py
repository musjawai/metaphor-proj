TEMPLATE_MAP = {
    "generate_queries": """
    You are a helpful assistant that generates search queiries based on user questions. You are given the following user question. Using this, \
    you must generate one search query. You must output just this search query.
    User Question: {user_question}
    """,
    'summarize_content_from_url': """
    You are a helpful assistant that generates a summary of around 250 words from a webpage. Focus on what tangible metrics. Summarize the following content: \
    {content}
    """, 
    'summarize_content_from_db': """
    You are a helpful assistant that generates a summary of around 100 words of a text. Summarize the following text: \
    {text}
    """,
}

def template_factory(template_id):
    template = TEMPLATE_MAP.get(template_id)
    if template is None:
        raise ValueError(
            f"Template does not exist. Enter one of {TEMPLATE_MAP.keys()}"
        )
    return template