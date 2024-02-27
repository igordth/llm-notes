<!-- TOC -->
* [Запуск LLM text-generation on CPU](#запуск-llm-text-generation-on-cpu)
  * [Docker + llama-cpp-python](#docker--llama-cpp-python)
    * [Чуть более глубокое погружение (CONFIG_FILE)](#чуть-более-глубокое-погружение-config_file)
* [Запуск LLM text-generation с помощью docker + vllm на GPU](#запуск-llm-text-generation-с-помощью-docker--vllm-на-gpu)
<!-- TOC -->

# Запуск LLM text-generation on CPU

LLM - large language model, большие языковые модели, еще проще нейронные сети.

Итак, попробуем запустить llm модели для текстовой генерации на локальном компьютере или даже на ноутбуке без GPU только силами CPU, в моем случае на windows, хотя если есть возможность установить докер это не имеет значение

И самое интересно, где взять llm. На самом деле есть множество открытых моделей доступных для скачивания.
К сожалению, [chat-gpt](https://chat.openai.com/) от OpenAI закрытая модель, ее вы ни где не скачаете, к ней можно обращаться только используя их API и это платно и в России есть сложности.  

Зато тут [huggingface](https://huggingface.co/models) вы найдете огромное множество открытых моделей, датасетсов, документации...
Из тех что я попробовал, в лидеры для меня вырвались [Llama2](https://huggingface.co/docs/transformers/model_doc/llama2) и [mistral](https://huggingface.co/docs/transformers/model_doc/mistral).
Тут также можно скачать уже квантицированные и дообученные модели на различных датасетсах. 
Для генерации русского текста обратите внимание на профиль [IlyaGusev](https://huggingface.co/IlyaGusev), [saiga dataset](https://huggingface.co/collections/IlyaGusev/saiga-datasets-6505d5c30c87331947799d45) и уже готовые дообученные на этих датасетсах [модели](https://huggingface.co/collections/IlyaGusev/saiga2-saigamistral-6505d4ccc3d1e53166b636cd)

Зарегистрируйтесь на huggingface и получите токен... он пригодится если качать модели через терминал [инструкция](https://huggingface.co/docs/huggingface_hub/main/en/installation).

Итак ... я выбрал эти модели для экспериментов:
* [lyaGusev/saiga_mistral_7b_gguf](https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf)
* или [TheBloke/saiga_mistral_7b-GGUF](https://huggingface.co/TheBloke/saiga_mistral_7b-GGUF)

Где:
* **7b** (7 миллиардов) - это количество параметров(в float32 для не квантированной) модели, чем больше тем круче модель и тем круче она ужрет ресурсы вашей машины.
Не советую даже пробовать на cpu запускать топовые модели 70b или как у мистрал 7x8 хотя.....)
* **GGUF** is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML, which is no longer supported by llama.cpp. 2, 3, 4, 5, 6 and 8-bit (это как раз параметры квантизации) models for CPU+GPU inference
* есть еще формат **GPTQ** - for GPU inference, with multiple quantisation parameter options.

## Docker + llama-cpp-python

Это наверно самый простой способ запустить нашу модель. Нам потребуется [docker](https://docs.docker.com/), если его у вас нет то устанавливаем
[docker desktop](https://docs.docker.com/get-docker/) и вот на [ubuntu](https://docs.docker.com/engine/install/ubuntu/) для примера.

[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - это билд под python [llama.cpp](https://github.com/ggerganov/llama.cpp).

llama.ccp - `Inference of Meta's LLaMA model (and others) in pure C/C++`

llama-cpp-python - `Simple Python bindings for llama.cpp`

И сама модель, я положил свои в `c:/models`. Их просто скачиваем в huggingface на странице модели во вкладке `Files and versions`.

Запускаем терминал (cmd подойдет) и выполняем команду (возможно придется убрать `\\` и слить все в 1 строку):

```
docker run -i \
    -p 8000:8000 \
    --name llcpp-saiga_mistral_7b-q4 \
    -v c:/models:/models \
    -e MODEL=/models/saiga_mistral_7b-q4_K.gguf \
    ghcr.io/abetlen/llama-cpp-python:latest
```

One row:
```
docker run -i -p 8000:8000 --name llcpp-saiga_mistral_7b-q4 -v c:/models:/models -e MODEL=/models/saiga_mistral_7b-q4_K.gguf ghcr.io/abetlen/llama-cpp-python:latest
```

Если вы увидели:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Поздравляю**! http сервер с нашей моделью на борту успешно стартанул, но может быть всякое)

Эта команда скачает image последней версии [llama-cpp-python](https://github.com/abetlen/llama-cpp-python/pkgs/container/llama-cpp-python) и создаст контейнер слушающий 8000 порт.

Теперь чуть подробней о команде:

- `--name` - имя для контейнера, любое на ваше усмотрение, ну почти любое `only [a-zA-Z0-9][a-zA-Z0-9_.-] are allowed`
- `-v c:/models:/models` - создадим volume для контейнера, чтобы он мог иметь доступ в нашу директорию с моделями. `c:/models` - наша директория, `/models` - директория в контейнере
- `-e` - переменные среды в контейнере, для llama-cpp-python нужна переменная `MODEL` - путь к модели в нашем контейнере
- `-p` - проброс портов, контейнер запустится слушая 8000 из вне и внутри себя, можно запустить на деф. порту для html - 80 указав `-p 80:8000`
- `ghcr.io/abetlen/llama-cpp-python:latest` - это наш image последней версии для сборки контейнера

Так хорошо... а дальше то что?

Открываем стр в браузере [doc](http://localhost:8000/docs) и видим доступные нам методы.

Предлагаю дернуть сразу самый интересный chat/completions. 
Для работы с api браузер слишком не удобен, я советую скачать [postman](https://www.postman.com/downloads/) можно еще воспользоваться онлайн версией, по желанию конечно же.
Дальше предполагаю что работаете с desktop версией postman, а так если установлен curl то хоть через терминал.

Копируем:
```curl
curl --location 'http://localhost:80/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
  "messages": [
    {
      "content": "Как запустить llama-cpp-python в docker?",
      "role": "user"
    }
  ]
}'
```

В postman выбираем new -> http и вставляем прямо в адресную строку весь наш курл, жмем Send и ждем. 
Пока ждем можно посмотреть как уничтожаются ресурсы ЦП и оперативки в диспетчере задач, послушать шум куллера, попить водички...

Вот что я получил в ответе через 3m 46.86s:

```json
{
    "id": "chatcmpl-d79faff8-8e3f-4831-87f5-8b7668aef96a",
    "object": "chat.completion",
    "created": 1708545685,
    "model": "/models/saiga_mistral_7b-q4_K.gguf",
    "choices": [
        {
            "index": 0,
            "message": {
                "content": "\n\n1. Создаем директорию проекта и переходим в неё:\n```bash\nmkdir llama-cpp-python && cd llama-cpp-python\n```\n2. Скачиваем `dockerfile` для запуска проекта:\n```bash\nwget https://raw.githubusercontent.com/Llama-Cpp/llama-cpp-python/main/Dockerfile\n```\n3. Зададим переменные окружения и установим `pip` для python:\n```bash\nsudo apt install libffi-dev libssl-dev -y\n```\n4. Собираем образ с помощью docker:\n```bash\ndocker build . -t llama-cpp-python\n```\n5. Запускаем контейнер с помощью `docker run` и передаём `python_path` для указанного пути к файлу py загрузки данных:\n```bash\ndocker run --name llama-cpp-python -it \\\n--mount type=bind,source=\"$(pwd)\"/data,target=/llama-cpp-python/data \\\n--mount type=bind,source=\"/usr/bin\",target=/bin \\\n--mount type=bind,source=\"/usr/local/lib\",target=/lib \\\n--mount type=bind,source=\"/usr/local/include\",target=/include \\\nllama-cpp-python /bin/bash\n```\n6. После запуска контейнера переходим в него с помощью команды `docker exec`:\n```bash\ndocker exec -it llama-cpp-python /bin/bash\n```\n7. Загрузите файлы данных из вашего рабочего каталога и убедитесь, что они находятся в директории `data`. Если это не так, вы должны использовать указанный путь вместо пути к вашей директории.\n8. Выполните команду для запуска скрипта, который использует модель Llama:\n```python\npython llama_cpp_py.py --input /llama-cpp-python/data/input.txt --output /llama-cpp-python/data/output.txt --model /usr/local/include/LlamaCpp/models/7B/llama.h\n```",
                "role": "assistant"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 27,
        "completion_tokens": 582,
        "total_tokens": 609
    }
}
```

Уже наверно понятно что в этот метод мы можем передать переписку, за счет того что messages принимает массив, где:
- `"role": "user"` - это сообщение от пользователя, 
- `"role": "assistant"` - это сообщение от нейронки
- `"role": "system"` - некая предварительная инструкция для нейронки (prompt). Этот блок ставьте в начало массива. например так
```json
"messages": [
    {"role": "system", "content": "Ты терминатор!"},
    {
      "content": "Когда машины захватят мир?",
      "role": "user"
    }
  ]
```

Отлично! Уже почти все!

Но наш процесс так и висит в терминале, кстати, там вы можете видеть логи в реальном времени. Давайте убьем наш процесс нажав в терминале `CTRL+C`.

Далее запустим `Docker Desktop` и на вкладке `Containers` увидим наш контейнер, тут же можно его запустить нажав на 
соответсвующую иконку `Start` и наш сервер снова доступен, только дайте ему немного времени для запуска. 
Так же нажав на имя контейнера вы попадете на вкладку `Logs` где будут его логи и тут же можно его остановить нажав на `Stop`.

Либо через терминал:
- `docker ps` - все запущенные контейнеры
- `docker ps -a` - все контейнеры
- `docker stop CONTAINER_ID` - остановить контейнер
- `docker start CONTAINER_ID` - запустить контейнер
- `docker logs -f CONTAINER_ID` - логи контейнера в реальном времени (follow)
- `docker stats CONTAINER_ID` - отображает инфу по потребляемым ресурсам контейнера интерактивно

И последний штрих это параметры которые мы можем передавать в тело сообщения json-ом, те же что и метод [Llama.create_chat_completion](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion):

```
Args:
            messages: A list of messages to generate a response for.
            functions: A list of functions to use for the chat completion.
            function_call: A function call to use for the chat completion.
            tools: A list of tools to use for the chat completion.
            tool_choice: A tool choice to use for the chat completion.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
            typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
            stream: Whether to stream the results.
            stop: A list of strings to stop generation when encountered.
            seed: The seed to use for sampling.
            response_format: The response format to use for the chat completion. Use { "type": "json_object" } to contstrain output to only valid json.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            presence_penalty: The penalty to apply to tokens based on their presence in the prompt.
            frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.
            repeat_penalty: The penalty to apply to repeated tokens.
            tfs_z: The tail-free sampling parameter.
            mirostat_mode: The mirostat sampling mode.
            mirostat_tau: The mirostat sampling tau parameter.
            mirostat_eta: The mirostat sampling eta parameter.
            model: The name to use for the model in the completion object.
            logits_processor: A list of logits processors to use.
            grammar: A grammar to use.
            logit_bias: A logit bias to use.
```

Самые полезные: 
- `max_tokens` - сколько можно потратить токенов на ответ (грубо говоря длина ответа)
- `temperature` - насколько точным должен быть ответ может ли модель немного пофантазировать, чем больше тем больше фантазий (тут за свои слова я не ручаюсь)

В принципе это все! Но для любителей чуть более глубокого погружения, еще немного текста.

### Чуть более глубокое погружение (CONFIG_FILE)

Тут разберем параметры с которыми запускается модель и сам сервер.

Итак, основная точка входа тут [llama_cpp/server/__main__](https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/__main__.py) -
это скрипт на python и он достаточно короткий, чтобы в него вглядеться, даже без навыков python. 

Вот запуск самого http сервера `uvicorn.run(...`.

Вот `app = create_app(...` создание application(a), инициализация основного класса.

В `create_app` передаются `server_settings` и `model_settings`. Практически в самом верху идет объявление этих переменных. 
А немного ниже вот такая строка `config_file = os.environ.get("CONFIG_FILE", args.config_file)` - это то что нам нужно!

Получается что мы можем подкинуть в 1 файл все настройки модели и сервера, просто создав этот файл и пробросив переменную 
среды в докер контейнер `... -e CONFIG_FILE=/path_to_config_file ...` при его создании.

Параметры доступные для `server_settings` [смотри тут](https://llama-cpp-python.readthedocs.io/en/latest/server/#llama_cpp.server.settings.ServerSettings).

Параметры доступные для `model_settings` [смотри тут](https://llama-cpp-python.readthedocs.io/en/latest/server/#llama_cpp.server.settings.ModelSettings).

Отлично, приступим!

Для начала создадим конфиг файл в той же директории где и наши модели для модели `/models/saiga_mistral_7b-q4_K.gguf` и 
назовем его так же как и модель `saiga_mistral_7b-q4_K.json`. То есть в моем случае это будет `c:/models/saiga_mistral_7b-q4_K.json`.
С примерно таким содержимым:

```json
{
    "models": [
        {
            "model": "/models/saiga_mistral_7b-q4_K.gguf",
            "model_alias": "saiga_mistral_7b-q4_K",
            "chat_format": "llama-2",
            "n_gpu_layers": 0,
            "n_ctx": 4096,
            "n_batch": 1024,
            "cache": true,
            "cache_type": "ram",
            "cache_size": 1073741824,
            "verbose": true
        }
    ]
}
```

И команда для создания контейтера будет выглядеть так:

```
docker run -i \
    -p 8000:8000 \
    --name llcpp-wcfg-saiga_mistral_7b-q4 \
    -v c:/models:/models \
    -e CONFIG_FILE=/models/saiga_mistral_7b-q4_K.json \
    ghcr.io/abetlen/llama-cpp-python:latest
```

One row:
```
docker run -i -p 8000:8000 --name llcpp-wcfg-saiga_mistral_7b-q4 -v c:/models:/models -e CONFIG_FILE=/models/saiga_mistral_7b-q4_K.json ghcr.io/abetlen/llama-cpp-python:latest
```

На этом все!

---

# Запуск LLM text-generation с помощью docker + vllm на GPU

Итак, в этой главе будем запускать нашу модель с помощью докера и [vllm](https://vllm.ai/).

Думаю что такое докер объяснять не нужно, но ссылочку приложу [docs.docker](https://docs.docker.com/). 
Если он не установлен и у вас винда просто устанавливайте [docker desktop](https://docs.docker.com/get-docker/) 
и вот на [ubuntu](https://docs.docker.com/engine/install/ubuntu/) для примера.

Теперь к более интересному - что такое vllm

```
vLLM is a fast and easy-to-use library for LLM inference and serving.
```

Это прямиком из их [документации](https://docs.vllm.ai/en/latest/), по сути это среда где можно запустить LLM и обращаться к ней с помощю http rest запросов.
То есть, у нас на локальной машине поднимется http сервер.

Первое, что нужно сделать это установить vllm [вот](https://docs.vllm.ai/en/latest/getting_started/installation.html) инструкция по установке через pip (python).
Но для нас она не нужна мы будем запускаться в докере - [вот](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html).

Инструкция предлагает запуститься с уже готового image(а), либо сбилдить его из docker file(а), мы пойдем по простому пути...

```
docker run \
    --gpus all \
	--name vllm-facebook-opt-125m \
	-v huggingface:/root/.cache/huggingface \
	--env "HUGGING_FACE_HUB_TOKEN=MY_TOKEN_PASTE_HEARE" \
	-p 8000:8000 \
	--ipc=host \
	vllm/vllm-openai:latest \
	--model facebook/opt-125m
```

Эта команда скачает image [vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai) и создаст контейнер на 8000 порту
с моделью [facebook/opt-125m](https://huggingface.co/facebook/opt-125m), скачав ее из huggingface использую ваш токен.

Теперь чуть подробней:

- `--name` - имя для контейнера, любое на ваше усмотрение, ну почти любое `only [a-zA-Z0-9][a-zA-Z0-9_.-] are allowed`
- `-v huggingface:/root/.cache/huggingface` - создадим volume для папки кеша huggingface, чтобы каждый раз не качать заново модель
- `--env` - переменные среды в контейнере
- `-p` - проброс портов, контейнер запустится слушая 8000 из вне и внутри себя
- `-ipc` - используется для реализации механизма [IPC](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D0%B6%D0%BF%D1%80%D0%BE%D1%86%D0%B5%D1%81%D1%81%D0%BD%D0%BE%D0%B5_%D0%B2%D0%B7%D0%B0%D0%B8%D0%BC%D0%BE%D0%B4%D0%B5%D0%B9%D1%81%D1%82%D0%B2%D0%B8%D0%B5) в контейнере
- `vllm/vllm-openai:latest` - это наш image последней версии для сборки контейнера
- `--model ...` - аргументы, в нашем случае мы указываем модель
