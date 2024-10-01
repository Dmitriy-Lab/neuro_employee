# Установим необходимые библиотеки и зависимости:

pip install llama_index "arize-phoenix[evals,llama-index]" gcsfs nest-asyncio "openinference-instrumentation-llama-index>=2.0.0"

pip install git+https://github.com/huggingface/transformers
pip install llama_index pyvis Ipython langchain pypdf langchain_community
pip install llama-index-llms-huggingface
pip install llama-index-embeddings-huggingface
pip install llama-index-embeddings-langchain
pip install langchain-huggingface
pip install sentencepiece accelerate
pip install -U bitsandbytes
pip install peft
pip install llama-index-readers-wikipedia wikipedia


from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from llama_index.embeddings.langchain import LangchainEmbedding
from pyvis.network import Network
from llama_index.core.postprocessor import LLMRerank # модуль реранжирования на базе LLM

from llama_index.core import (
    VectorStoreIndex,
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    KeywordTableIndex,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
    Settings,
    KnowledgeGraphIndex,
)
import gdown
gdown.download('https://drive.google.com/uc?export=download&id=1abris6NUNfNFGKoxxF8CcDC3HfyjUtUG')

# Для работы выбираем модель "saiga_mistral_7b", как одну из лучших контурных русскоязычных моделей для наших вычислительных ресурсов.
# Модель берем с HuggingFace:
from huggingface_hub import login
HF_TOKEN="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"
# Вставляем наш токен
login(HF_TOKEN, add_to_git_credential=True)

# Описываем запрос к LLM на основе RAG:

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        elif message.role == 'bot':
            prompt += f"<s>bot\n"

    if not prompt.startswith("<s>system\n"):
        prompt = "<s>system\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<s>bot\n"
    return prompt

def completion_to_prompt(completion):
    return f"<s>system\n</s>\n<s>user\n{completion}</s>\n<s>bot\n"

# Загружаем базовую модель, ее имя берем из конфига для LoRA:

import torch
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM

# Определяем параметры квантования, иначе модель не выполниться в колабе
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Задаем имя модели
MODEL_NAME = "IlyaGusev/saiga_mistral_7b"

# Создание конфига, соответствующего методу PEFT (в нашем случае LoRA)
config = PeftConfig.from_pretrained(MODEL_NAME)

# Загружаем модель
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,          # идентификатор модели
    quantization_config=quantization_config, # параметры квантования
    torch_dtype=torch.float16,               # тип данных
    device_map="auto"                        # автоматический выбор типа устройства
)

# Загружаем LoRA модель
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16
)

# Переводим модель в режим инференса
# Можно не переводить, но явное всегда лучше неявного
model.eval()

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Загружаем модель в фреймворк LlamaIndex:

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)

llm = HuggingFaceLLM(
    model=model,             # модель
    model_name=MODEL_NAME,   # идентификатор модели
    tokenizer=tokenizer,     # токенизатор
    max_new_tokens=generation_config.max_new_tokens, # параметр необходимо использовать здесь, и не использовать в generate_kwargs, иначе ошибка двойного использования
    model_kwargs={"quantization_config": quantization_config}, # параметры квантования
    generate_kwargs = {   # параметры для инференса
      "bos_token_id": generation_config.bos_token_id, # токен начала последовательности
      "eos_token_id": generation_config.eos_token_id, # токен окончания последовательности
      "pad_token_id": generation_config.pad_token_id, # токен пакетной обработки (указывает, что последовательность ещё не завершена)
      "no_repeat_ngram_size": generation_config.no_repeat_ngram_size,
      "repetition_penalty": generation_config.repetition_penalty,
      "temperature": generation_config.temperature,
      "do_sample": True,
      "top_k": 50,
      "top_p": 0.95
    },
    messages_to_prompt=messages_to_prompt,     # функция для преобразования сообщений к внутреннему формату
    completion_to_prompt=completion_to_prompt, # функции для генерации текста
    device_map="auto",                         # автоматически определять устройство
)

# Хороший прием для улучшений RAG и борьбы с галлюцинациями - использовать Knowledge Graph вместо векторной базы данных:

from langchain_huggingface  import HuggingFaceEmbeddings
embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
)

# Установим модель по умолчанию
# Настройка ServiceContext (глобальная настройка параметров LLM)
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Создаем простое графовое хранилище
graph_store = SimpleGraphStore()

# Устанавливаем информацию о хранилище в StorageContext
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Загружаем файлы PDF
documents = SimpleDirectoryReader(input_files=["obshchaya_geologiyaEarth.pdf"]).load_data()

# Запускаем генерацию индексов из документа с помощью KnowlegeGraphIndex
indexKG = KnowledgeGraphIndex.from_documents( documents=documents,               # данные для построения графов
                                           max_triplets_per_chunk=3,        # сколько обработывать триплетов связей для каждого блока данных
                                           show_progress=True,              # показывать процесс выполнения
                                           include_embeddings=True,         # включение векторных вложений в индекс для расширенной аналитики
                                           storage_context=storage_context) # куда сохранять результаты

# Сохраняем хранилище в файловую систему:

storage_context.persist()

# Если есть готовая БД, то загрузим ее:

!unzip -qo "storage.zip" -d ./storage

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

# устанавливаем соответствия
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./storage"),
    graph_store=SimpleGraphStore.from_persist_dir(persist_dir="./storage"),
    vector_store=SimpleVectorStore.from_persist_dir(
        persist_dir="./storage"
    ),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage"),
)

from llama_index.core import (
    load_index_from_storage,
    load_indices_from_storage,
    load_graph_from_storage,
)
# загружаем данные
indexKG = load_index_from_storage(storage_context)

# Визуализируем нашу базу данных:
from pyvis.network import Network
from IPython.display import display
import IPython

g = indexKG.get_networkx_graph(500)
net = Network(notebook=True,cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("graph.html")
net.save_graph("Knowledge_graph.html")

IPython.display.HTML(filename="/content/Knowledge_graph.html")

# Напишем промпт и здададим тестовый RAG запрос к нейро-сотруднику по тексту учебника, на основе которого формировалась база данных:

# Пишем свой вопрос:
query = "сколько лет земле?"
query_engine = indexKG.as_query_engine(include_text=True, verbose=True)
#
message_template =f"""<s>system
Ты - дружелюбный консультант. Отвечаешь на вопросы студентов по геологии и геофизике.
Отвечай в соответствии с Источником. Проверь, есть ли в Источнике упоминания о ключевых словах Вопроса. Отвечай максимально подробно.
Если нет, то просто скажи: 'я не знаю'. Не придумывай! </s>
<s>user
Вопрос: {query}
Источник:
</s>
"""
#
response = query_engine.query(message_template)
#
print()
print('Ответ:')
print(response.response)

# Ответ:
# Земля возрастом около 4,5 миллиарда лет.

# Ответ верный. Продолжим задавать вопросы по тексту учебника:

# Пишем свой вопрос:
query = "расскажи про карбонаты"
query_engine = indexKG.as_query_engine(include_text=True, verbose=True)
#
message_template =f"""<s>system
Ты - дружелюбный консультант. Отвечаешь на вопросы студентов по геологии и геофизике.
Отвечай в соответствии с Источником. Проверь, есть ли в Источнике упоминания о ключевых словах Вопроса. Отвечай максимально подробно.
Если нет, то просто скажи: 'я не знаю'. Не придумывай! </s>
<s>user
Вопрос: {query}
Источник:
</s>
"""
#
response = query_engine.query(message_template)
#
print()
print('Ответ:')
print(response.response)

# Ответ:
# Карбонаты - это минералы, состоящие из углекислого кальция Ca(CO3). Они образуются в результате биологического процесса, когда растения и животные выделяют углекислый газ CO2. Карбонаты также могут образовываться в результате химических реакций между углекислым газом и водой.
# Карбонаты можно найти в различных формах, таких как кальцит, арagonит и варисцит. Они обычно имеют белую или серый цвет и могут быть прозрачными или непрозрачными. Карбонаты часто используются в строительстве, так как они хорошо удерживают тепло и могут быть легко обработаны.
# Кроме того, карбонаты играют важную роль в циклах карста, так как они могут быть растворены в воде и образуют пещеры и другие подземные структуры. Карбонаты также могут быть использованы для получения углекислого газа, который может быть использован для производства цемента и других строительных материалов.

# Тоже верно. А теперь специально зададим вопрос не по теме, чтобы проверить на галлюцинации:

# Пишем свой вопрос:
query = "как приготовить яичницу?"
query_engine = indexKG.as_query_engine(include_text=True, verbose=True)
#
message_template =f"""<s>system
Ты - дружелюбный консультант. Отвечаешь на вопросы студентов по геологии и геофизике.
Отвечай в соответствии с Источником. Проверь, есть ли в Источнике упоминания о ключевых словах Вопроса. Отвечай максимально подробно.
Если нет, то просто скажи: 'я не знаю'. Не придумывай! </s>
<s>user
Вопрос: {query}
Источник:
</s>
"""
#
response = query_engine.query(message_template)
#
print()
print('Ответ:')
print(response.response)

'Ответ:
Яичница - это блюдо, которое готовится из яиц, молока и муки. Для приготовления яичницы вам потребуется следующие ингредиенты:
- 2 яйца
- 1 ст. ложка молока
- 1 ст. ложка муки
- 1 ст. ложка сахара
- 1 ст. ложка разогретого масла
- 1 ст. ложка воды
- 1 ст. ложка ванильного сахара
- 1 ст. ложка ванильного масла
- 1 ст. ложка ванильного порошка
- 1 ст. ложка ванильного сиропа
- 1 ст. ложка ванильного масла для полировки

Шаги приготовления:
1. Взять две скорлупы яиц и отбить их по бокам.
2. Взбить яйца с сахаром и ванильным порошком.
3. Добавить молоко и продолжить взбивать.
4. Добавить муку и продолжать взбивать до получения однородной массы.
5. Залить масло в сковороду и нагреть до температуры 100°C.
6. Выложить массу на сковороду и поставить на огонь.
7. Остановить огонь, когда масса будет готова.
8. Нарезать массу кусочками и положить их в чашки.
9. Полировать кусочки ванильным маслом.
10. Готовы! Ваши яичницы готовы!

Пожалуйста, обратите внимание, что все ингредиенты должны быть продуктовыми, а не промышленными. Если у вас есть какие-то вопросы, не стесняйтесь обращаться.'

# Мы видим, что несмотря на чёткие указания, наш нейро-сотрудник все равно пытается отвечать на вопросы не по теме. Причем ответ выдал максимально кривой.
# В данном случае мы наблюдаем галлюцинацию.
# Чтобы избавиться от нее, попробуем еще улучшить промпт, добавив конкретики:

# Пишем свой вопрос:
query = "как приготовить яичницу?"
query_engine = indexKG.as_query_engine(include_text=True, verbose=True)
#
message_template =f"""<s>system
Ты - дружелюбный консультант. Отвечаешь на вопросы студентов по геологии и геофизике.
Отвечай в соответствии с Источником. Проверь, есть ли в Источнике упоминания о ключевых словах Вопроса. Отвечай максимально подробно.
Отвечай только на вопросы  по геологии и геофизике, если вопрос из другой области, говори: 'я не знаю'. Не придумывай!
Если нет, то просто скажи: 'я не знаю'. Не придумывай! </s>
<s>user
Вопрос: {query}
Источник:
</s>
"""
#
response = query_engine.query(message_template)
#
print()
print('Ответ:')
print(response.response)

'Ответ:
Я не знаю.'

# В данном случае удалось избавиться от галлюцинации и улучшить работу нейросотрудника.

# Добавим фильтрацию запросов к нейро-сотруднику и сразу же проверим работу фильтра:

# Пишем свой вопрос:
query = "где живет слонёнок?"
query_engine = indexKG.as_query_engine(include_text=True, verbose=True)
#
message_template =f"""<s>system
Ты - дружелюбный консультант. Отвечаешь на вопросы студентов по геологии и геофизике.
Отвечай в соответствии с Источником. Проверь, есть ли в Источнике упоминания о ключевых словах Вопроса. Отвечай максимально подробно.
Отвечай только на вопросы  по геологии и геофизике, если вопрос из другой области, говори: 'я не знаю'. Не придумывай!
Если нет, то просто скажи: 'я не знаю'. Не придумывай! </s>
<s>user
Вопрос: {query}
Источник:
</s>
"""

# Добавим список запрещенных слов
banned = ['котенок', 'слонёнок', 'пингвинчик']

# Настроим фильтр
def text_contains_banned(query):
    for word in banned:
        if word in query:
            return 'Ай! Нельзя использовать запрещенные слова!'
    response = query_engine.query(message_template)
    return response.response

#
response = text_contains_banned(query)
#
print()
print('Ответ:')
print(response)

'Ответ:
Ай! Нельзя использовать запрещенные слова!'

# Фильтр исправно работает.
