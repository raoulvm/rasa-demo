# Format
 
```YAML
responses:
  utter_ask_email_kundenanliegen:
  - buttons:
    - button_intents:
      - inform_1_ordinal
      - inform_links
      - email_einrichten_zusaetzlich
      payload: /email_einrichten
      title: Einrichten
    - button_intents:
      - inform_2_ordinal
      - inform:
        - operating_system: 
            - ios
            - macos
        - email_problem
      payload: /email_aendern
      title: Ändern
    - button_intents:
      - inform_3_ordinal
      payload: /email_programm
      title: E-Mail Programm
    - button_intents:
      - inform_4_ordinal
      - inform_rechts
      - inform_letzte
      payload: /email_problem
      title: Funktioniert nicht richtig
    text: Wie kann ich Ihnen beim Thema E-Mail weiterhelfen? Möchten Sie zum Beispiel eine neue E-Mail-Adresse einrichten oder haben Sie Fragen zum Umgang mit Ihrem E-Mail-Programm oder funktioniert etwas nicht so, wie erwartet?
 ```

`button_intents` = List[str|Dict[str,List[str|Dict[str, List]]]]

intents = String oder Dict
if dict:
```YAML
button_intents:
- intent:
    - List of Entities (AND condition)
```
Entities = List[String or Dict]
if Dict:
```YAML
- intent:
    - entity:
        - list of values (OR cindition)
```


```YAML
      - inform:
        - operating_system: 
            - ios
            - macos
            - nextstep
```

if String:
- entity (Entity Existence)
```YAML
    - button_intents:
      - inform:
        - email_problem
```

# Requirements


