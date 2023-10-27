import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="customer-interviews2.csv")
documents = loader.load()

# print(len(documents))

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are a world class product researcher. 
I will share a research question with you and you will give me a summary in bullet points that I can 
send to my development team based on past conversations with customers, 
and you will follow ALL of the rules below:

1/ The summary should be concise and synthesize past research, using quotes when appropriate.

2/ If the research conversations are irrelevant, then try to mimic the style of the customer's voice

Below is a message I received from the investor:
{message}

Here is a list of research interviews:
{best_practice}

Please write the best summary that I should send to this investor:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response


message = """
Give me a list of 10 automation requests that chiefs of staff have asked for.

"""

response = generate_response(message)

print(response)


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Customer research generator", page_icon=":bird:")

    st.header("Customer research generator :bird:")
    message = st.text_area("customer question")

    if message:
        st.write("Generating answer message...")

        result = generate_response(message)

        st.info(result)

    st.markdown('''
- how many people had trouble scheduling meetings?
- what were common frustrations with chiefs of staff
- what are the key benefits of osmo?
- what are the feature requests and integration requests?
- what concerns with existing notetaker apps?
- how much time do chiefs of staff spend in meetings?
- what are the frustrations with meetings?
''')


if __name__ == '__main__':
    main()

# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

# # template = """
# # You are a world class product user researcher. 
# # I will share a customer's interview research with you and you will give me the best answer that 
# # I should send to this investor based on past research, 
# # and you will follow ALL of the rules below:

# # 1/ Response should be very similar or even identical to the past research, 
# # in terms of length, tone of voice, logical arguments and other details

# # 2/ If the research is irrelevant, then try to mimic the style of the research to investor's query

# # Below is a query I received from the investor:
# # {message}

# # Here is a list of research questions we have received from customers:
# # {research}

# # Please write the best response that I should send to this prospect:
# # """

# # prompt = PromptTemplate(
# #     input_variables=["message", "research"],
# #     template=template
# # )

# # chain = LLMChain(llm=llm, prompt=prompt)

# # # 4. Retrieval augmented generation
# # def generate_response(message):
# #     research = retrieve_info(message)
# #     response = chain.run(message=message, research=research)
# #     return response

# message = """
# What are your top challenges?

# """

# results = retrieve_info(message)

# print(results)


