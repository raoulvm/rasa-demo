# Format
 
```YAML
responses:
  utter_ask_playground_install_info:
  - text: Which would you like to do first?
    buttons:
    - payload: /get_started_playground
      title: Try out Rasa in the online Rasa Playground
      button_intents:
        - enter_data:
          - product: "playground"
    - payload: /install_rasa
      title: Install Rasa on your own computer
      
    - payload: '/trigger_response_selector{{"retrieval_intent":"chitchat/ask_whatisrasa"}}'
      title: Find out more about Rasa
      button_intents:
        - how_to_get_started
 ```

`button_intents` = List[str|Dict[str,List[str|Dict[str, List]]]]

# Using intents without entities
If button_intents contains list elements that are simple strings, those are matched against the user intent without entity comparison:
```YAML
    - payload: '/trigger_response_selector{{"retrieval_intent":"chitchat/ask_whatisrasa"}}'
      title: Find out more about Rasa
      button_intents:
        - how_to_get_started
```
In this case, if the user triggers the `how_to_get_started` intent after the button utterance, the policy will replace that with the playload of 
`/trigger_response_selector{{"retrieval_intent":"chitchat/ask_whatisrasa"}}`, i.e. the intent will be `trigger_resonse_selector` with the entity `retrieval_ intent` set to the value of `chitchat/ask_whatisrasa`.

If multiple intents are given in the list, they are treated as OR condition.

# Using intents with entities

Entities are either checked for existence (if a list without values) or checked for specific values (if dict entries):

intents = String oder Dict  

if dict:
```YAML
button_intents:
- intent:
    - List of Entities (AND condition)
```
If more then one entity is given for one intent, it is treated as AND condition: te intent must have all the entities included to trigger.

Entities = List[String or Dict[str, list]]

As an entity can only have one value, a list of values are treated as an OR condition.
```YAML
- intent:
    - entity:
        - list of values (OR cindition)
```


In this example, the intent `enter_data` is only replaced if the entity `product` has the value `playground`. Other then TED and MemoizationPolicy it is working on the text value, not on the featurized value. 
```YAML
      button_intents:
        - enter_data:
          - product: "playground"
```
If the values are a list, they are treated as alternatives.


