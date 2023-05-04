import abc
import dataclasses
import functools
import logging
from enum import Enum
from functools import reduce
from typing import Union, List, Dict, Tuple, Callable, TypeVar, cast, Any, Optional


# todo 在sqlalchemy、tortoise等orm框架中验证拼接出来的SQL可行性
# todo python Union语法糖可能不兼容低版本（eg:python3.9）,而且当Union在第一层时，使用新语法很丑，语义不明确
# todo 入参没有强制校验，实际应该报错处理

def string_concat(
        strs: Union[List[Union[str, int]], Tuple[Union[str, int]], str, int],
        separator:str = None,
        boundary: Tuple[str, str] = None,
        strict_mode: bool = False
):
    """
    元素拼接
    支持任意一个元素，都将会转化为字符串进行拼接，支持自定义分割符，自定义包裹方式

    :param strs: 元素列表或单个元素
    :param separator: 各元素拼接的分隔符字符串
    :param boundary: 一个元组，用于指定前后包裹方式
    :param strict_mode: 严格模式，是否严格区分字符串和整数，严格模式下所有字符串会特别注明
    :return: 一个拼接字符串

    >>> params = {"strs": [1, 2, "3"], "separator": ",", "boundary": ("(", ")"), "strict_mode": True}
    >>> string_concat(**params)
    "(1,2,'3')"

    >>> params = {"strs": [1, 2, 3], "separator": ".", "boundary": ("(", ")"), "strict_mode": True}
    >>> string_concat(**params)
    '(1.2.3)'
    """
    if strs is None:
        return ""

    strs = strs if isinstance(strs, (list, tuple)) else [strs]

    # 严格模式下，字符串也会被转义
    if strict_mode:
        strs = map(lambda x: x if isinstance(x, int) else f"\'{x}\'", strs)

    ret = f"{'' if separator is None else str(separator)}".join(map(str, strs))

    if boundary:
        return boundary[0] + ret + boundary[1]

    return ret

class OrderEnum(Enum):
    DESC = "DESC"
    ASC = "ASC"

class ConditionConcatEnum(Enum):
    AND = " AND "
    OR = " OR "

    _AND = "__and"
    _OR = "__or"

    @property
    def and_(self) -> str:
        return self._AND.value

logger = logging.getLogger()

# 使用空格拼接的条件类型
space_separator_conditions = {"__limit", "__offset", "__group_by", "__order_by"}

# (权重,操作类型,操作名称)
# 为操作设置权重，方便对每一个流程块中的条件字符串排序，权重越重，拼接时越靠前，权重越轻，拼接时越靠后。
# 为操作设置类型，同类型的操作使用 {1: 前缀操作， 2: 后缀操作， 3： 特殊操作}
conditions_sorted_weight = [
    (10, 1, "__limit"), (15, 1, "__offset"), (20, 1, "__order_by"), (25, 1, "__group_by"),
    (30, 3, "__extra"), (35, 1, "__is_null"), (40, 1, "__al"), (45, 1, "__rl"), (50, 1, "__ll"), (55, 1, "__in"), (60, 1, "__not_in"),
    (65, 1, "__ne"), (70, 1, "__gte"), (75, 1, "__lte"), (80, 1, "__ge"), (80, 1, "__lt"), (85, 1, "__e"), (90, 1, "__end")
]

def condition_concat(x: Tuple[str, str], y: Tuple[str, str], separator: ConditionConcatEnum) -> Tuple[str, str]:
    """
    条件拼接操作
    两个参数之间的的拼接方式， 用于OR或者AND之间的条件拼接

    :param x: 一个条件元组 ("__gt", "age>10")
    :param y: 一个条件元组 ("__gt", "age>10")
    :param separator: 拼接条件
    :return:
    """
    if x[0] not in space_separator_conditions and y[0] not in space_separator_conditions:
        return y[0], string_concat(strs=[x[1], y[1]], separator=separator.value)
    else:
        return y[0], string_concat(strs=[x[1], y[1]], separator=" ")

and_operation_concat: Callable = functools.partial(condition_concat, separator=ConditionConcatEnum.AND)
or_operation_concat: Callable = functools.partial(condition_concat, separator=ConditionConcatEnum.OR)

def _in(field: str, value: Union[List[Union[str, int]], Tuple[Union[str, int]]], **kwargs) -> str:
    """
    SQL在列表中处理

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串
    """
    strict_mode = kwargs.get("strict_mode", True)
    return f"{field} IN {string_concat(value, separator=',', strict_mode=strict_mode, boundary=('(', ')'))}"

def _not_in(field: str, value: Union[List[Union[str, int]], Tuple[Union[str, int]]], **kwargs) -> str:
    """
    SQL在列表中处理

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串
    """
    strict_mode = kwargs.get("strict_mode", True)
    return f"{field} NOT IN {string_concat(value, separator=',', strict_mode=strict_mode, boundary=('(', ')'))}"

def _gt(field: str, value: int, **kwargs) -> str:
    """
    SQL大于(>)处理

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串
    """
    return f"{field}>{value}"

def _lt(field: str, value: int, **kwargs) -> str:
    """
    SQL小于(<)处理

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串
    """
    return f"{field}<{value}"

def _e(field: str, value: int, **kwargs) -> str:
    """
    SQL等于(=)处理

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串

    todo
        1. 字符串类型和整数类型没有区分
    """
    strict_mode = kwargs.get("strict_mode", True)
    return f"{field}={string_concat(value, strict_mode=strict_mode)}"

def _gte(field: str, value: int, **kwargs) -> str:
    """
    SQL大于等于(>=)处理

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串
    """
    return f"{field}>={value}"

def _lte(field: str, value: int, **kwargs) -> str:
    """
    SQL小于等于(<=)处理

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串
    """
    return f"{field}<={value}"

def _ne(field: str, value: Union[str, int], **kwargs) -> str:
    """
    SQL不等于(!=)处理

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串
    """
    strict_mode = kwargs.get("strict_mode", True)
    return f"{field}!={string_concat(value, strict_mode=strict_mode)}"

def _ll(field: str, value: str, **kwargs) -> str:
    """
    SQL左模糊匹配处理

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串
    """
    value = f"%%{value}"

    strict_mode = kwargs.get("strict_mode", True)
    return f"{field} LIKE {string_concat(value, strict_mode=strict_mode)}"

def _rl(field: str, value: str, **kwargs) -> str:
    """
    SQL右模糊匹配处理

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串
    """
    value = f"{value}%%"

    strict_mode = kwargs.get("strict_mode", True)
    return f"{field} LIKE {string_concat(value, strict_mode=strict_mode)}"

def _al(field: str, value: str, **kwargs) -> str:
    """
    SQL全模糊匹配处理

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串
    """
    value = f"%%{value}%%"

    strict_mode = kwargs.get("strict_mode", True)
    return f"{field} LIKE {string_concat(value, strict_mode=strict_mode)}"

def _is_null(field: str, value: bool, **kwargs) -> str:
    """
    SQL空值处理
    支持布尔值(True, False, 1, 0)

    :param field: 字段名
    :param value: 字段值
    :return: str 一个拼接字符串

    todo
        1. 类型定义校验不够完善，缺少布尔值校验
    """
    return f"{field} IS NULL" if value else f"{field} IS NOT NULL"

def _group_by(value: Union[str, List[str], Tuple[str]], **kwargs) -> str:
    """
    SQL分组操作处理
    支持使用单值，多值定义

    :param value: 一个字符串、字段元组或者列表
    :return: str 一个拼接字符串
    """
    return f"GROUP BY {string_concat(value, separator=',')}"

def _order_by(value: Union[str, List[Union[Dict[str, Union[bool, OrderEnum]]]]], **kwargs) -> str:
    """
    SQL排序操作处理
    支持单值（单值时默认值为倒序 DESC）、多值排序，支持使用布尔值(True, False, 0, 1)或者枚举值（DESC, ASC, desc, asc）去定义指定字段的排序方式

    :param value: 一个列表，元素为一个字典或者一个二值元祖
    :return: str 一个拼接字符串

    >>> value = [("created_at", True), ("deleted_at", "desc")]
    >>> _order_by(value)
    "ORDER BY created_at ASC, deleted_at DESC"

    >>> value = [{"created_at": True, "deleted_at": "desc"}]
    >>> _order_by(value)
    "ORDER BY created_at ASC,deleted_at DESC"

    >>> value = "created_at"
    >>> _order_by(value)
    "CREATED_AT ASC"

    todo
        1. 类型定义校验不够完善，缺少二值元组校验
        2. 错误捕获不够优雅
    """
    if isinstance(value, (List, Tuple)):
        values = [(v[0], OrderEnum.DESC.value if v[1] else OrderEnum.ASC.value) for v in value]
    elif isinstance(value, dict):
        values = [(k, OrderEnum.DESC.value if o else OrderEnum.ASC.value) for k, o in value.items()]
    elif isinstance(value, str):
        values = [(value, OrderEnum.DESC.value)]
    else:
        raise ValueError("order by 解析参数类型错误！")

    clauses = [f"{v[0]} {v[1].upper()}" for v in values]

    return f"ORDER BY {string_concat(clauses, separator=',')}"

def _limit(value: Union[int, List[int], Tuple[int]], **kwargs) ->str:
    """
    SQL数量限制操作处理
    支持使用单值，多值定义

    :param value: int|str单值， 二值元组或者二值列表
    :return: str 一个拼接字符串
    >>> value = 1
    >>> _limit(value)
    "LIMIT 1"

    >>> value = [1, 2]
    >>> _limit(value)
    "LIMIT 1,2"

    todo
        1. 类型定义校验不够完善，缺少二值元组校验
    """
    strict_mode = kwargs.get("strict_mode", True)
    value = value if isinstance(value, (List, Tuple)) else [value]
    return f"LIMIT {string_concat(value, separator=',', strict_mode=strict_mode)}"

def _offset(value: int, **kwargs) ->str:
    """
    SQL位移操作处理

    :param value: int
    :return: str 一个拼接字符串
    """
    strict_mode = kwargs.get("strict_mode", True)
    return f"OFFSET {string_concat(value, strict_mode=strict_mode)}"

def _end(field: str, value: Union[List[Union[str, int]], Tuple[Union[str, int]], int, str], **kwargs) ->str:
    """
    特殊流程处理函数
    简化操作，不使用任何后缀，直接相等， 支持 相等(=)、在列表(in) 操作

    :param field: 字段名
    :param value: 字段值，单值，元组或者列表操作
    :return: str 一个拼接字符串
    """
    if isinstance(value, (List, Tuple)):
        return _in(field, value, **kwargs)
    return _e(field, value, **kwargs)

def _or(value: List[Tuple[str, str]], **kwargs) -> str:
    """
    OR流程处理函数
    使用SQL中的OR处理一个流程块

    :param value: 流程块列表
    :return: str 一个拼接字符串

    >>> value = [("__end", "age=1"), ("__is_null", "created_at IS NOT NULL")]
    >>> _or(value)
    "(age=1 OR created_at IS NOT NULL)"

    >>> value = [("__end", "age=1"), ("__is_null", "created_at IS NOT NULL"), ("__limit", "LIMIT 10")]
    >>> _or(value)
    "(age=1 OR created_at IS NOT NULL LIMIT 10)"
    """
    value = list(filter(lambda x: x[1], value))
    if not value:
        return ""

    strs = string_concat(reduce(or_operation_concat, value)[1])

    # 优化结构，当有多个元素时，才需要括号包裹，单个元素不需要括号包裹，对整体结构来说更加清晰
    if len(value) > 1:
        strs = string_concat(strs, boundary=("(", ")"))
    return strs

def _and(value: List[Tuple[str, str]], **kwargs) -> str:
    """
    AND流程处理函数 (默认流程处理函数）
    使用SQL中的AND处理一个流程块

    :param value: 流程块列表
    :return: str 一个拼接字符串

    >>> value = [("__end", "age=1"), ("__is_null", "created_at IS NOT NULL")]
    >>>_and(value)
    "(age=1 AND created_at IS NOT NULL)"

    >>> value = [("__end", "age=1"), ("__is_null", "created_at IS NOT NULL"), ("__order_by", "ORDER BY created_at DESC")]
    >>> _and(value)
    "(age=1 AND created_at IS NOT NULL ORDER BY created_at DESC)"

    >>> value = ["age=1", "created_at IS NOT NULL", "ORDER BY created_at DESC", ("__extra", "(select * from user)")]
    >>> _and(value)
    "(age=1 AND created_at IS NOT NULL ORDER BY created_at DESC AND (select * from user))"
    """
    # fixme 遗留问题，AND流程中空值时，条件全部丢弃，但是参数未丢弃的错误
    # if not all([v[0] for v in value]):
    #     return ""

    value = list(filter(lambda x: x[1], value))
    if not value:
        return ""

    strs = string_concat(reduce(and_operation_concat, value)[1])

    # 优化结构，当有多个元素时，才需要括号包裹，单个元素不需要括号包裹，对整体结构来说更加清晰
    if len(value) > 1:
        strs = string_concat(strs, boundary=("(", ")"))
    return strs

def _extra(value: str, **kwargs) -> str:
    """
    特殊流程处理函数
    支持拼接任意自定义SQL, 所有内容都转为字符串

    :param value: 自定义SQL字符串
    :return: str
    """
    return str(value)

# 后缀操作处理，添加这些后缀时，会进行对应的处理
suffix_condition_map = {
    "__end": _end,
    "__gt": _gt,
    "__lt": _lt,
    "__e": _e,
    "__gte": _gte,
    "__lte": _lte,
    "__ne": _ne,
    "__ll": _ll,
    "__rl": _rl,
    "__al": _al,
    "__in": _in,
    "__not_in": _not_in,
    "__is_null": _is_null,
    "__extra": _extra,
}

# 前缀操作处理，对于流程块的处理
prefix_condition_map = {
    "__limit": _limit,
    "__offset": _offset,
    "__group_by": _group_by,
    "__order_by": _order_by,
    "__or": _or,
    "__and": _and,
    "__extra": _extra,
}

# 特殊操作处理，该工具会将一个字典当做一个流程块，递归到每一个条件进行解析，用于处理值为字典的操作。
# 例如，{"__offset": "id"}, offset操作的值为一个字符串，可以直接解析，
# 但是{"__order_by": {"name": True}}, order_by操作的值可以为一个字典，需要特殊这类操作
extra_dict_condition_map = {
    "__order_by": _order_by
}

suffix_condition_syntactic_sugar_map = {
    ">=": "__gte",
    "<=": "__lte",
    "~": "__ne",
    "%": "__al",
}

def compatible_old_syntactic_sugar(s):
    FLAG_NEG = "~"
    FLAG_GE = ">="
    FLAG_LE = "<="
    FLAG_LIKE = "%"
    for flag in [FLAG_LIKE, FLAG_LE, FLAG_GE, FLAG_NEG]:
        flag_len = len(flag)
        if len(s) > flag_len and s[-flag_len:] == flag:
            return flag, s[:-flag_len]

    return None, s

def get_prefix_and_suffix(flag: str) -> Tuple[str, str]:
    """
    解析条件字符串，获取字段的后缀标识

    :param flag: 字符+后缀标识的字符串 "__gt", "__lt", "__gte", "_lte", "__ne", "__ll", "__rl", "__al", "__in", "__not_in"
    :return: Tuple[str, str] (字段，字段标识)

    >>> flag = "age__gt"
    >>> get_prefix_and_suffix(flag)
    ("age", "__gt")
    """
    # 解析字段名称，取出原始字段与对应操作标识
    sep = str(flag).rfind("__")
    if sep > 0 and flag[sep:] and flag[sep+2:]:
        return flag[:sep], flag[sep:]

    return flag, "__end"

def meta_condition_map(flag: str, value: Any, **kwargs) -> Tuple[Callable, str, Optional[str], Any]:
    """
    通用流程映射
    所有的操作都会被分配给指定的处理函数，并打包参数，等待调用

    :param flag: 所支持的所有操作的前缀或后缀标识字符串 "__end", "__gt", "__lt", "__e", "__gte", "_lte", "__ne", "__ll", "__rl", "__al", "__in", "__not_in",
            "__limit", "__offset", "__group_by", "__order_by", "__and", "__or", "__extra"
    :param value: 参数值
    :return: Tuple[Callable, str, str, Any] (条件处理函数，条件标识， 字段， 字段值）

    >>> params = ("age__gt", 100)
    >>> meta_condition_map(*params)
    (_gt, "__gte", "name", 100)

    >>> params = ("__order_by", {"created_at": True})
    >>> meta_condition_map(*params)
    (_order_by, "__order_by", None, {"created_at": True})
        ...
    >>> params = ("__offset", 10)
    >>> meta_condition_map(*params)
    (_offset, "__offset", None, 10)
    """
    # 流程块操作相应处理
    if flag in prefix_condition_map:
        return prefix_condition_map[flag], flag, None, value

    # 兼容旧语法糖
    sugar_suffix, prefix = compatible_old_syntactic_sugar(s=flag)
    if suffix := suffix_condition_syntactic_sugar_map.get(sugar_suffix):
        return suffix_condition_map[suffix], suffix, prefix,  value

    # 后缀操作对应处理
    prefix, suffix = get_prefix_and_suffix(flag)
    return suffix_condition_map[suffix], suffix, prefix,  value

@dataclasses.dataclass
class MetaCondition(object):
    """
    一个条件的解析模型，包含这个条件的大部分信息

    flag: 解析前的条件字段，如__limit, age__gt
    value: 解析前条件的值，如 [1, 2], 100
    flag_sign: 解析后条件的特殊标志，如 __order_by, __gt, __in
    flag_func: 解析后条件的对应处理函数，如 __group_by: Callable, __in: Callable
    field: 解析后条件的真实字段，如 name, age, None: NoneType
    """
    flag: str = None
    value: Any = None
    flag_sign: str = None
    flag_func: Callable = None
    flag_value: Any = None
    field: str = None


class HookFunc(object):
    """
    单条件处理钩子
    支持对每一个条件进行自定义处理，默认启用SQL预处理钩子，并收集参数
    """

    @staticmethod
    def __default_hook(meta_cond: MetaCondition, **kwargs):
        """
        默认空钩子
        可以用于处理找不到指定钩子时的额外操作

        :params meta_cond: MetaCondition类型，当前处理条件的所有可用信息
        :params kwargs: 自定义参数

        :return: 统一响应 meta_cond, kwargs
        """
        # 给定一个钩子报错等级，指定未找到预期钩子时是跳过还是报错, 默认报错。
        if kwargs.get("hook_error", True):
            raise TypeError("预期钩子未找到！")

        return meta_cond, kwargs

    @staticmethod
    def sql_preprocess_hook(meta_cond: MetaCondition, **kwargs):
        """
        SQL预处理钩子
        支持参数替换 %s，并收集参数

        :param meta_cond: MetaCondition类型，当前处理条件的所有可用信息
        :params query_params: 参数收集容器
        :params strict_mode: 关闭拼接操作严格模式，严格模式下字符串"'"也会被转义，而所有替换字符"%s"都是字符串，无需转义

        :return: 统一响应 meta_cond, kwargs
        """

        # SQL预处理时，需要收集参数的条件类型
        collection_params_operations = {
            "__end", "__gt", "__lt", "__e", "__gte", "_lte", "__ne", "__ll", "__rl", "__al", "__in", "__not_in",
            "__limit", "__offset"
        }

        # SQL防注入预处理,参数采集
        if meta_cond.flag_sign in collection_params_operations:
            # todo 多值与单值无法明确判断
            kwargs.setdefault("query_params", [])
            kwargs["query_params"].extend(meta_cond.value if isinstance(meta_cond.value, (list, tuple)) else [meta_cond.value])
            kwargs["strict_mode"] = False
            flag_value = ["%s" for _ in range(len(meta_cond.value))] if isinstance(meta_cond.value, (list, tuple)) else "%s"

            meta_cond.flag_value = flag_value

        return meta_cond, kwargs

    @staticmethod
    def null_handle_hook(meta_cond: MetaCondition, **kwargs):
        """
        空值处理钩子
        支持自定义参数值为空时的处理， 0、False被视为有效空值，拼接时正常处理，
        如 age=0, ORDER BY created_at DESC ,
        如 [], {}, (), None之类的被视为无效空值，此条件失效，不再参与拼接

        :param meta_cond: MetaCondition类型，当前处理条件的所有可用信息
        :return: 统一响应 meta_cond, kwargs

        """
        # fixme break时，参数捕获错误，在and拼接流程中，其中一个条件失效则本流程全部条件失效，or拼接流程中，其中一个条件失效不影响其他条件
        def _default_flag_func(**kwargs):
            return None

        if meta_cond.value or meta_cond.value == 0 or meta_cond.value is False:
            return meta_cond, kwargs
        elif meta_cond.value is None:
            meta_cond.flag_sign = ""
            meta_cond.flag_value = ""
        else:
            meta_cond.flag_sign = None
            meta_cond.flag_value = None
            meta_cond.flag_func = _default_flag_func

        return meta_cond, kwargs

    @classmethod
    @abc.abstractmethod
    def execute(cls, meta_cond: MetaCondition, **kwargs):
        """
        自定义钩子执行入口
        支持对每一个条件进行自定义处理，默认启用SQL预处理钩子，并收集参数

        :params meta_cond: MetaCondition类型，当前处理条件的所有可用信息
        :params user_hook: 是否启用钩子的开关，默认启用钩子
        :params hook_error: 是否捕获钩子报错，指定未找到预期钩子时是跳过还是报错, 默认报错
        :params hooks: List 钩子列表， 默认启用SQL预处理钩子 hooks=["sql_preprocess_hook"]
        :params kwargs: 自定义参数

        :return: 统一响应 meta_cond, kwargs
        """
        # 默认启用钩子
        if not kwargs.get("use_hook", True):
            return meta_cond, kwargs

        # 默认启用SQL预处理钩子
        kwargs.setdefault("hooks", ["null_handle_hook", "sql_preprocess_hook"])
        # 支持使用多个钩子，执行顺序按照默认顺序
        for hook in kwargs.get("hooks", []):
            execute_hook_func = getattr(cls, hook, cls.__default_hook)
            meta_cond, kwargs = execute_hook_func(meta_cond, **kwargs)

        return meta_cond, kwargs

def execute_hook(meta_cond: MetaCondition, **kwargs) -> Tuple[MetaCondition, Dict]:
    """
    执行自定义钩子
    自定义钩子错误抛出异常

    :param meta_cond: MetaCondition类型，当前处理条件的所有可用信息
    :return:

    """
    # 自定义处理钩子
    hook: HookFunc = kwargs.get("hook_func", HookFunc)
    if isinstance(hook, HookFunc) and hasattr(hook, "execute"):
        raise TypeError("钩子类型错误！")
    return hook.execute(meta_cond, **kwargs)


def general_process_layer(flag: str, value: Any, **kwargs) -> Tuple[str, Any]:
    """
    通用流程处理层
    获取每一个条件对应的处理函数，并执行调用，响应最终结果，本层可以获取每一个条件最终函数执行之前的所有元数据，可以在本层对所有条件进行全局处理，如SQL防注入预处理,参数采集

    :param flag: 所支持的所有操作的前缀或后缀标识字符串 [
            "__end", "__gt", "__lt", "__e", "__gte", "_lte", "__ne", "__ll", "__rl", "__al", "__in", "__not_in",
            "__limit", "__offset", "__group_by", "__order_by", "__and", "__or", "__extra"
            ]
    :param value: 参数值
    :return: str 一个拼接字符串

    >>> params = ("age__gt", 100)
    >>> general_process_layer(*params)
    ("__gt", "age<100")

    >>> params = ("user.id__in", [1, 2, 3])
    >>> general_process_layer(*params)
    ("__in", "user.id IN (1,2,3)")

    >>> params = ("__order_by", {"created_at": True})
    >>> general_process_layer(*params)
    ("__order_by", "ORDER BY created_at ASC")
        ...
    >>> params = ("__offset", 10)
    >>> general_process_layer(params)
    ("__limit", "OFFSET 10")
    """
    # 通用操作处理
    flag_func, flag_sign, field, flag_value = meta_condition_map(flag, value)
    meta_cond = MetaCondition(flag=flag, flag_func=flag_func, flag_sign=flag_sign, field=field, flag_value=flag_value, value=value)

    # 自定义处理钩子
    meta_cond, kwargs = execute_hook(meta_cond, **kwargs)

    return meta_cond.flag_sign, meta_cond.flag_func(field=meta_cond.field, value=meta_cond.flag_value, **kwargs)

def condition_parse(flag: str, value: Dict, **kwargs) -> Tuple[str, Any]:
    """
    条件解析
    会将字典结构视为一个“块”，每一个块中可以包含若干个条件与值的映射{condition: value}，解析每一个条件，拼接为字符串

    :param flag: 所支持的所有操作的前缀或后缀标识字符串
        -> flag in [
            "__end", "__gt", "__lt", "__e", "__gte", "_lte", "__ne", "__ll", "__rl", "__al", "__in", "__not_in",
            "__limit", "__offset", "__group_by", "__order_by", "__and", "__or", "__extra"
            ]
    :param value: 参数值
    :return: Tuple[str, str] (条件标识, 参数拼接结果)

    >>> params = {"flag": "__end", "value": {"name": "ttt", "age__in": [1, 2]}
    >>> condition_concat(params)
    ("__end", "name='ttt' AND age IN (1,2)")

    >>> params = {"flag": "__gt", "value": {"age__lt": "10"}
    >>> condition_concat(params)
    ("__gt", "age<10")

    >>> params = {"flag": "__order_by", "value": {"created_at": True}}
    >>> condition_concat(params)
    ("__order_by", "ORDER BY created_at ASC")

    >>> params = {"flag": "__and", "value": {"age__lt": 10, "name": "ttt"}
    >>> condition_concat(params)
    ("__and", "age<10 AND name='ttt'")

    >>> params = {"flag": "__or", "value": {"age__lt": 10, "name": "ttt"}
    >>> condition_concat(params)
    ("__or", "(age<10 OR name='ttt'")

    >>> params = {"flag": "__extra", "value": "group by name,id"}
    >>> condition_concat(params)
    ("__extra", "group by name,id")
    """
    # 特殊操作处理，用于解析值为字典的操作
    if flag in extra_dict_condition_map:
        return general_process_layer(flag, value, **kwargs)

    conditions_str = []
    for sub_flag, sub_value in value.items():
        # 当字段值为流程块（字典）时，进入流程块，递归处理每一条条件参数
        if isinstance(sub_value, Dict):
            sub_condition = condition_parse(sub_flag, sub_value, **kwargs)
            conditions_str.append(sub_condition)
            continue

        # 普通对应参数处理
        flag_sign, flag_value = general_process_layer(sub_flag, sub_value, **kwargs)
        conditions_str.append((flag_sign, flag_value))

    return general_process_layer(flag, conditions_str, **kwargs)

def db_condition_parse(cond: Dict, **kwargs) -> Tuple[Union[str, Any], List[Any]]:
    """
    SQL查询条件拼接
    根据传递的格式化数据，拼接出指定的查询参数，自身并不执行SQL校验或查询操作。

    1. 支持全等(=)、不等(!=)、大于(>)、大于等于(>=)、小于(<)、小于等于(<=)、左模糊匹配(LIKE)、右模糊匹配(LIKE)、全模糊匹配(LIKE)、
    在列表(IN)、不在列表(NOT IN)、空值判断(IS NULL, IS NOT NULL)、分页限制(LIMIT)、位移(OFFSET)、排序(ORDER BY)、分组(GROUP BY)、
    与操作(AND)、或操作(OR)、自定义操作(eg: 支持在任意位置添加任意数量的自定义操作)

    2. 支持自定义全局钩子，对每一个条件自定义处理函数，
        2.1 默认开启SQL预处理钩子，支持参数替换(%s)，参数采集

    3. 支持字段类型精确匹配的索引操作，（eg: 在SQL预处理中，由于存在参数替换，可能不支持精确索引匹配）

    4. 支持空值判断，目前只有 0、False被认为有效空值，其他空值如[], (), {}, None 在AND流程中舍弃当前上下文流程块全部条件，在OR流程中舍弃当前条件

    5. 兼容旧解析中的四个语法糖，>=, <=, ~, %

    todo:
        1. 需要给一些操作提供默认值
        2. 在最终条件处理函数中增加一个自定义回调函数，便于局部自定义处理, 空值处理，可能每一个操作都不一样 （eg: 局部钩子可以通过全局钩子实现，在全局钩子中返回自定义处理函数）
        3. __gt之类的操作嵌套复杂操作，比如 WHERE user_id = 123456789 AND fs_id > (上次查询结果中最后一条记录的id值) ORDER BY fs_id LIMIT 300000;
            3.1. 后期可能单字段多级嵌套处理， (eg: 当前可用__extra特殊操作直接拼接)
        4. 所有的值都被解析为字符串，无法充分利用索引
            4.1. 当前采用SQL预处理替换参数的防注入方案，无法解决精确利用索引问题
        5. 兼容一些常见的语法糖，比如 ~, !=
        6. 对于增、删、改等其他操作的支持， 对于更新需求，比如，update user set value=value+1 where id=1
        7. 关于双重否定的优化，如 id__not_in []，实际想要的效果应该是 “查全部”，因为所有id都不为空
            7.1 当前实现为正常拼接，空值条件会被舍弃，（eg: 实现 “查全部”的效果可以通过钩子函数来实现，或者自定义字段，当前无法局部实现，只能通过__extra实现）
        8. 定义一个操作，可以拼接一大堆相同的条件，暂时没想到怎么实现，或许可以使用自定义局部钩子做到。

    :param cond: 查询参数字典
    :return: (str, List) 一个拼接字符串，一个参数列表（默认添加SQL预处理钩子，并采集参数）

    >>> cond = {"name": "ttt"}
    >>> db_condition_parse(**cond)
    (("name='ttt'), ['ttt'])

    >>> cond = {"age__in": [1, 2], "__limit": 10, "order_by": [("created_at", True), ("updated_at", "DESC")]}
    >>> db_condition_parse(**cond)
    (("age IN (%s, %s) LIMIT 10 ORDER BY created_at ASC, updated_at DESC"), [1, 2, 10])

    >>> cond = {"__extra": "select * from user where id=1"}
    >>> db_condition_parse(**cond)
    (("select * from user where id=1"), [])

    >>> cond = {"__or": {
                "name__ll": "你好",
                "is_deleted__is_null": True",
                "__or": {
                    "__extra": "(任意一段字符串)",
                    "age__gte": 100,
                    "1__extra": "(任意一段字符串)",
                    "__limit": [1, 10],
                    "2__extra": "(另一段字符串)"
                    }
                }
            }
    >>> db_condition_parse(**cond)
    ((name LIKE %%%s OR is_deleted IS NULL OR ((任意一段字符串) OR age>=%s OR (任意一段字符串) LIMIT %s,%s (另一段字符串))),['你好', 100, 1, 10])

    """
    origin_sql = "WHERE "
    query_params = []

    if not cond:
        return "", []

    concat_result = condition_parse(ConditionConcatEnum.AND.and_, cond, query_params=query_params, **kwargs)[1]

    return origin_sql + concat_result, query_params


if __name__ == '__main__':
    # conds = {
        # "user.id__in": ["你好", "1", 2],
        # "1__extra": "(aaaa)",
        # "user_id>=": 0,
        # "id<=10": 100,
        # "name%": "你好",
        # "id~": 1,
        # "4__extra": "你好啊啊",
        # "name": [],
        # "5__extra": "",
        # "__limit": 10
        # "lawyer": "张三",
        # "lawyer__ne": "李四",
        # "2__extra": "(bbbbb)",
        # "__extra": "11111",
        # "__and": {
        #     "age__gt": 1,
        #     "__order_by": {
        #         "name": False,
        #         "age": 0
        #     },
        #     "name": "name",
        # },
        # "age__gt": 10,
        # "user_name": "ttt",
        # "age": 1,
        # "is_deleted__is_null": False,
        # "__group_by": "name",
        # "__or": {
        #     "name": 0,
        #     "age__in": [1,2,3],
        #     "__or": {
        #         "user.id__in": [1,2,3],
        #         "employ.id__is_null": True,
        #         "__extra": "(select 1 as extra_str)",
        #         "__limit": [1,2],
        #         "__order_by": {
        #             "created_at": "ASC",
        #             "deleted_at": "desc",
        #         },
        #         "__or": {
        #             "__order_by": "lawyer_censor",
        #         },
        #         "__group_by": ["name", "id"],
        #     },
        #     "__extra": "ifnull(0, 1)",
        #     "__limit": (1,2),
        #     "__group_by": ["id", "name"]
        # },
        # "name__al": "你好",
        # "name__rl": "你好",
        # "name__ll": "你好",
        # "__limit": 1,
        # "__offset": 10,
        # "__order_by": [("created_at", "desc"), ("updated_at", "asc")],
        # "3__extra": "group by name"
    # }

    # conds = {
    #     "age__in": [1, 2],
    #     "__limit": 10,
    #     "__order_by": "created_at"
    # }

    conds = {"__or": {
                "name__ll": "你好",
                "is_deleted__is_null": True,
                "__or": {
                    "__extra": "(任意一段字符串)",
                    "age__gte": 100,
                    "1__extra": "(任意一段字符串)",
                    "__limit": [1, 10],
                    "2__extra": "(另一段字符串)"
                    }
                }
            }
    sql, params = db_condition_parse(conds)
    print(sql)
    print(params)
