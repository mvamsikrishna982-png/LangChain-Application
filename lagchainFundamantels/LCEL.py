from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key = os.getenv('GROQ_API_KEY')
)

Prompt = ChatPromptTemplate.from_messages(
    [
        ('system','You are professinal ML tutor.'),
        ('human','Explian this {topic} in few sentences .')
    ]
)

parser = StrOutputParser()

########### BASIC CHAIN CALL ####################################
chain = Prompt|llm|parser

result = chain.invoke({'topic':'fine tuning'})
print("=============================.invoke()========================")
print(result)

print("=============================.stream()========================")
for chk in chain.stream({'topic':'fine tuning'}):
    print(chk, end = "", flush = True)

print("=============================.batch()========================")

result = chain.batch([
    {'topic':'bias and variance trade off.'},
    {'topic':'underfitting'},
    {'topic':'overfitting'}
]
)
print(result)

####################### RunnablePassThrough ##################

chain = Prompt|RunnablePassthrough()|llm|parser #RunnablePassthrough do nothing pass the input as it is.
result = chain.invoke({'topic':'fine tuning'})
print(result)

####################### RunnableLambda ##################

p = PromptTemplate(

    input_variables = ['topic'],
    template = 'you are an ML expert, explain this topic {topic} in few sentences.'
)

def fun(prompt):

    return prompt.text + " say Jai Hind!! after at the end of your response."


chain = p|RunnableLambda(fun)|llm|parser #RunnableLambda wraps a functions

print(chain.invoke({'topic':'batch size'}))


###################### RunnableParallel ##############################


tPromt = PromptTemplate(
    template = 'Expain this topic:{topic} in telugu. and at the end say sasank and anand are erripukulu'
)

chain = Prompt|llm|parser


translate = tPromt|llm|parser

seq_chain = RunnableParallel(
    english  = chain,
    telugu = translate
)

result = seq_chain.invoke({'topic':'Supervised learning'})

print(result)

"""
{
'english': 'Supervised learning is a type of machine learning where the model is trained on labeled data,
meaning the data is already categorized or classified. The goal is to teach the model to map inputs to outputs
based on the labeled examples, so it can make predictions on new, unseen data. The model learns from the labeled
data by adjusting its parameters to minimize the error between its predictions and the actual labels, allowing it
to make accurate predictions on future data. This type of learning is commonly used for tasks such as image classification,
text classification, and regression analysis.', 'telugu': 'పర్యవేక్షించషించబడిన అభ్యాసం (సూపర్\u200cవైజ్డ్ లెర్నింగ్) అనేది యంత్ర
అభ్యాసంలో ఒక రకమైన అభ్యాస పద్ధతి, ఇక్కడ మెషీన్ లెర్నింగ్ అల్గారిథమ్\u200cకు ఇప్పటికే లేబుల్ చేయబడిన డేటాను అందించడం ద్వారా 
అభ్యాసం చేయబడుతుంది. ఈ లేబుల్స్ సాధారణంగా డేటా యొక్క కావలసిన ప్రవర్తన లేదా ఫలితాన్ని సూచిస్తాయి.\n\nపర్యవేక్షించబడిన అభ్యాసంలో,
అల్గారిథమ్ అందించిన డేటా నుండి నమూనాలు మరియు సంబంధాలను గుర్తించడానికి ప్రయత్నిస్తుంది మరియు కొత్త, అందుబాటులో లేని డేటాకు 
లేబుల్\u200cలను అంచనా వేయడానికి ఉపయోగించగల మోడల్\u200cను నిర్మించడం లక్ష్యం.\n\nపర్యవేక్షించబడిన అభ్యాసం యొక్క ఉదాహరణలు:\n\
* చిత్రాలను వర్గీకరించడం (ఉదాహరణకు, కుక్కలు vs. పిల్లులు)\n* వాక్యాలను వర్గీకరించడం (ఉదాహరణకు, సానుకూలంగా vs. ప్రతికూలంగా)\n* 
వ్యక్తుల ఆదాయాన్ని అంచనా వేయడం వారి వయస్సు, విద్య మరియు ఇతర లక్షణాల ఆధారంగా\n\nససాంక్ మరియు ఆనంద్ ఎర్రిపుకులు.'
}   
"""