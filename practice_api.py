import os
from openai import OpenAI
import requests
from dotenv import load_dotenv

# load env
load_dotenv()

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY')
    base_url="https://api.deepseek.com"
)


#pubmed api

def search_pubmed(query, max_result=5):
    """
    search pubmed paper
    """
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    params = {
        "db":"pubmed",
        "term": query,
        "retmax":max_result,
        "retmode":"json"
    }

    try:
        response = requests.get(url=url, params=params)
        response.raise_for_status()
        data = response.json()

        pmids = data.get('esearchresult', {}).get('idlist', [])
        return {
            "success": True,
            "pmids": pmids,
            "count":len(pmids)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def fetch_pubmed_details(pmids):
    url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    parameters = {
        "dbfrom": "pubmed",
        "id":pmid,
        "retmode":"json",
        "cmd": "prlinks"
    }

    try:
        response = requests.get(url, parameters)
        response.raise_for_status()
        data = response.json()

        links = []
        linksets = data.get('linksets', [])
        if linksets:
            idurllist = linksets[0].get('idurlist', [])
            if idurllist:
                objurls = idurlist[0].get('objurls', [])
                for obj in objurls:
                    links.append({
                        "provider": obj.get('provider', {}).get('name','Unknown'),
                        "url": obj.get('url', '')
                    })

        return {
            "success": True,
            "pmid": pmid,
            "full_text_links": links
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

#agent tool

tools = [
    {
        "type":"function",
        "function":{
                    "name": "search_pubmed",
                    "description":"search pubmed database for scienfic articles" }
    }
]


#AI agent
def run_agent(user_query):
    messages = [
        {
            "role":"system",
            "content":"you are a helpful medical research assistant."
        },
        {
            "role": "user","content": user_query
        }
    ]

    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model='deepseel-chat',
            message=messages,
            tools=tools
            tempereture=0.7
        )
    assistant_message = response.choicese[0].message
    message.append(assistant_message)

    if assistant_message.tools_call:
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.lode(tool_call.function.arguments)

            function_to_call = available_functions.get(function_name)
            if function_to_call:
                function_response = function_to_call(**function_args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id":tool_call.id,
                        "name":function_name,
                        "content":json.dumps(function_respons, ensure_ascii=False)

                    }
                )
            else:
                print(f"{function_name}not find")

        else:

            return assistant_message.content
    return "timeout"

if __name__ == "__main__":
    while True:
        user_input = input("your qeustion: ").strip()

        if user_input.lower() in ['quit','exit']:
            print('googbay!')
            break
        if user_input:
            run_agent(user_input)
            
