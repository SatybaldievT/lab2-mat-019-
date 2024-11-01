**README**

**Графовые операции с помощью Actix Web**

Этот проект реализует набор операций с графами, используя Actix Web для построения RESTful API. API предоставляет следующие методы для работы с графами:

*   **POST /generate_graph**: генерирует новый граф по заданным параметрам.
*   **POST /check_table**: проверяет равенство двух автоматов по таблице.
*   **POST /check_membership**: проверяет принадлежность строки языку, заданному автоматом.
*   **GET /get_path**: возвращает случайный путь в графе.

### Установка и запуск

1.  Установите Rust и Cargo на свой компьютер.
2.  Клонируйте репозиторий и перейдите в папку с файлами проекта.
3.  Выполните команду `cargo run` для запуска сервера.

### Использование API

**POST /generate_graph**

*   **Запрос**: JSON-объект с полями `width`, `height`, `pr_of_break_wall` и `num_of_finish_edge`.
*   **Ответ**: JSON-объект с полем `graph`, представляющим сгенерированный граф.

Пример запроса:
```json
{
  "width": 10,
  "height": 10,
  "pr_of_break_wall": 20,
  "num_of_finish_edge": 2
}
```
```bash
curl -Uri http://localhost:8080/generate_graph -Method Post -Header @{ "Content-Type" = "application/json" } -Body '{"width": 1, "height": 1, "pr_of_break_wall": 0, "num_of_finish_edge": 1}'
```

**POST /check_table**

*   **Запрос**: JSON-объект с полями `main_prefixes`, `complementary_prefixes`, `suffixes` и `table`.
*   **Ответ**:  String "1" или "0", обозначающим равенство двух автоматов.

Пример запроса:
```json
  {
    "main_prefixes": "e N",
     "complementary_prefixes": "S W E NN NS NW NE", 
     "suffixes": "e NN N",
     "table": "011111011011011111011111111"
  }
```
```bash
curl -Uri http://localhost:8080/generate_graph -Method Post -Header @{ "Content-Type" = "application/json" } -Body '{"main_prefixes": "e N","complementary_prefixes": "S W E NN NS NW NE", "suffixes": "e NN N","table": "011111011011011111011111111"}'
```

**POST /check_membership**

*   **Запрос**: строка, принадлежность которой нужно проверить.
*   **Ответ**: String "1" или "0", обозначающим принадлежность строки языку.

Пример запроса:
```bash
curl -Uri http://localhost:8080/check_membership -Method Post -Header @{ "Content-Type" = "application/json" } -Body 'NNSWSNNSN'
```

**GET /get_path**

*   **Запрос**: нет.
*   **Ответ**: JSON-объект с полем `path`, представляющим случайный путь в графе.

Пример запроса:
```bash
curl -Uri http://localhost:8080/get_graph -Method Get     
```

**GET /get_graph**

*   **Запрос**: нет.
*   **Ответ**: JSON-объект графа формате
*   ```
     {
    "nodes": [
            {
                "id": 1,
                "value": 1,
                "isFinal": true,
                "neighbors": [
                    {
                        "neighbor": 1,
                        "symbol": "N"
                    },
                    {
                        "neighbor": 0,
                        "symbol": "S"
                    },
                    {
                        "neighbor": 1,
                        "symbol": "W"
                    },
                    {
                        "neighbor": 1,
                        "symbol": "E"
                    }
                ]
            },
            {
                "id": 0,
                "value": 0,
                "isFinal": false,
                "neighbors": [
                    {
                        "neighbor": 1,
                        "symbol": "N"
                    },
                    {
                        "neighbor": 0,
                        "symbol": "S"
                    },
                    {
                        "neighbor": 0,
                        "symbol": "W"
                    },
                    {
                        "neighbor": 0,
                        "symbol": "E"
                    }
                ]
            }
        ]
    }
    ```
и создает  графическое представление  действующего графа на сервере 

![get_graph](https://github.com/user-attachments/assets/2b4ea225-dc83-4c7a-8261-bfa9f7288046)



Пример запроса:
```bash
curl -Uri http://localhost:8080/get_graph -Method Get     

```


