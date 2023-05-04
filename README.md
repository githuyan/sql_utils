### 一个类Django-orm的SQL拼接工具，用于便捷的拼接SQL条件

```python
conditions = {
    "age__gt": 100,
    "__limit": (10, 20)
}
db_condition_parse(conditions)

"age>100 LIMIT 10,20"
```