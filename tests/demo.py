from sql_utils.orm import _e, _not_in


a = _not_in("id", [])

print(a)
