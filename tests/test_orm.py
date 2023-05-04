import pytest
from typing import List, Tuple, Union

from sql_utils.orm import string_concat, condition_concat, ConditionConcatEnum, _in, _not_in, _gt, _lt, _e, _gte, \
    _lte, _ll, _rl, _al, _is_null, _group_by, _order_by, OrderEnum, _limit, _offset, _end, _or, _and, _extra, \
    get_prefix_and_suffix, meta_condition_map, _ne, general_process_layer, condition_parse, \
    MetaCondition, HookFunc, db_condition_parse


class TestStringConcat:
    @pytest.mark.parametrize(
        "strs, separator, boundary, strict_mode, expected",
        [
            ([1, 2, 3], ".", None, False, "1.2.3"),
            (["hello", "world"], "-", None, False, "hello-world"),
            ([1, "2", 3], ".", ("(", ")"), False, "(1.2.3)"),
            ([1, "2", 3], ".", None, True, "1.'2'.3"),
            ([1, 2, 3], "-", ("[", "]"), True, "[1-2-3]"),
            ([1, 2, 3], None, None, False, "123"),
            ([], "", None, False, ""),
            (1, ".", None, False, "1"),
            ("hello", "-", None, False, "hello"),
            ((1, 2, 3), ".", None, False, "1.2.3"),
            ((1, "2", 3), ".", ("(", ")"), False, "(1.2.3)"),
            (["hello", 2], "-", None, False, "hello-2"),
            ([1, 2, 3], "-", ("(", ")"), True, "(1-2-3)"),
            ([1, 2, 3], ".", ("[", "]"), True, "[1.2.3]"),
            ([1, "2", 3], None, None, True, "1'2'3"),
        ],
    )
    def test_string_concat(
        self,
        strs: Union[List[str | int], Tuple[str | int], str, int],
        separator: str,
        boundary: Tuple[str, str],
        strict_mode: bool,
        expected: str,
    ):
        assert string_concat(strs, separator, boundary, strict_mode) == expected


class TestConditionConcat:
    def test_normal(self):
        x = ("__gt", "age>10")
        y = ("__lt", "age<1")
        res = condition_concat(x, y, ConditionConcatEnum.AND)
        assert res == ("__lt", "age>10 AND age<1")

    def test_no_space_separators(self):
        x = ("__gt", "age>10")
        y = ("__lt", "age<1")
        res = condition_concat(x, y, ConditionConcatEnum.OR)
        assert res == ("__lt", "age>10 OR age<1")

    def test_group_by(self):
        x = ("__gt", "age>10")
        y = ("__group_by", "GROUP BY name")
        res = condition_concat(x, y, ConditionConcatEnum.AND)
        assert res == ("__group_by", "age>10 GROUP BY name")

    def test_limit_offset(self):
        x = ("__offset", "OFFSET 10")
        y = ("__limit", "LIMIT 20")
        res = condition_concat(x, y, ConditionConcatEnum.AND)
        assert res == ("__limit", "OFFSET 10 LIMIT 20")


class TestIn:
    def test_in_with_list(self):
        field = "id"
        value = [1, 2, "3"]
        expected = "id IN (1,2,'3')"
        assert _in(field, value) == expected

    def test_in_with_tuple(self):
        field = "id"
        value = (1, 2, "3")
        expected = "id IN (1,2,'3')"
        assert _in(field, value) == expected


class TestNotIn:
    def test_not_in_with_list_of_integers(self):
        assert _not_in("id", [1, 2, 3]) == "id NOT IN (1,2,3)"

    def test_not_in_with_empty_list(self):
        assert _not_in("id", []) == "id NOT IN ()"

    def test_not_in_with_list_of_one_element(self):
        assert _not_in("id", [4]) == "id NOT IN (4)"

    def test_not_in_with_list_of_one_string_element(self):
        assert _not_in("name", ["Charlie"]) == "name NOT IN ('Charlie')"

    def test_not_in_with_list_of_mixed_types(self):
        assert _not_in("id", [1, "2", 3.0]) == "id NOT IN (1,'2','3.0')"

    def test_not_in_with_keyword_arguments(self):
        assert _not_in("id", [1, 2], table="users") == "id NOT IN (1,2)"


class TestGT:
    def test_gt(self):
        assert _gt("age", 10) == "age>10"
        assert _gt("salary", 10000) == "salary>10000"
        assert _gt("num", 0) == "num>0"
        assert _gt("amount", -100) == "amount>-100"


class TestLT:
    def test_lt(self):
        assert _lt("age", 10) == "age<10"
        assert _lt("salary", 10000) == "salary<10000"
        assert _lt("num", 0) == "num<0"
        assert _lt("amount", -100) == "amount<-100"


class TestE:
    def test_e_with_int_value(self):
        assert _e("age", 10) == "age=10"
        assert _e("salary", 10000) == "salary=10000"
        assert _e("num", 0) == "num=0"
        assert _e("amount", -100) == "amount=-100"

    def test_e_with_str_value(self):
        assert _e("name", "Alice") == "name='Alice'"
        assert _e("city", "New York") == "city='New York'"
        assert _e("country", "China") == "country='China'"


class TestGte:
    def test_gte_with_int_value(self):
        assert _gte("age", 10) == "age>=10"
        assert _gte("salary", 10000) == "salary>=10000"
        assert _gte("num", 0) == "num>=0"
        assert _gte("amount", -100) == "amount>=-100"

    # def test_gte_with_str_value(self):
    #     with pytest.raises(TypeError):
    #         _gte("name", "Alice")
    #     with pytest.raises(TypeError):
    #         _gte("city", "New York")
    #     with pytest.raises(TypeError):
    #         _gte("country", "China")


class TestNe:
    def test_int_value(self):
        assert _ne("age", 10) == "age!=10"

    def test_str_value(self):
        assert _ne("name", "John") == "name!='John'"

    # def test_empty_field(self):
    #     with pytest.raises(TypeError):
    #         _ne(None, 10)

    # def test_invalid_value(self):
    #     with pytest.raises(TypeError):
    #         _ne(None, "ten")


class TestLteFunction:
    def test_lte_with_valid_input(self):
        result = _lte("age", 10)
        assert result == "age<=10"

        result = _lte("height", 1.8)
        assert result == "height<=1.8"

        result = _lte("name", "John")
        assert result == "name<=John"

    # def test_lte_with_invalid_input(self):
    #     with pytest.raises(TypeError):
    #         _lte("age", "ten")
    #
    #     with pytest.raises(TypeError):
    #         _lte("name", {"value": "John"})


class TestLlFunction:
    def test_ll_with_valid_input(self):
        result = _ll("name", "hello")
        assert result == "name LIKE '%%hello'"

        result = _ll("address", "New York")
        assert result == "address LIKE '%%New York'"

        result = _ll("description", "a%b%c")
        assert result == "description LIKE '%%a%b%c'"

    # def test_ll_with_invalid_input(self):
    #     with pytest.raises(TypeError):
    #         _ll("age", 20)
    #
    #     with pytest.raises(TypeError):
    #         _ll("name", {"value": "hello"})


class TestRlFunction:
    def test_rl_with_valid_input(self):
        result = _rl("name", "hello")
        assert result == "name LIKE 'hello%%'"

        result = _rl("address", "New York")
        assert result == "address LIKE 'New York%%'"

        result = _rl("description", "a%b%c")
        assert result == "description LIKE 'a%b%c%%'"

    # def test_rl_with_invalid_input(self):
    #     with pytest.raises(TypeError):
    #         _rl("age", 20)
    #
    #     with pytest.raises(TypeError):
    #         _rl("name", {"value": "hello"})


class TestAlFunction:
    def test_al_with_valid_input(self):
        result = _al("name", "hello")
        assert result == "name LIKE '%%hello%%'"

        result = _al("address", "New York")
        assert result == "address LIKE '%%New York%%'"

        result = _al("description", "a%b%c")
        assert result == "description LIKE '%%a%b%c%%'"

    # def test_al_with_invalid_input(self):
    #     with pytest.raises(TypeError):
    #         _al("age", 20)
    #
    #     with pytest.raises(TypeError):
    #         _al("name", {"value": "hello"})


class TestIsNull:
    def test_is_null(self):
        assert _is_null("is_deleted", True) == "is_deleted IS NULL"
        assert _is_null("is_deleted", False) == "is_deleted IS NOT NULL"
        assert _is_null("is_deleted", 1) == "is_deleted IS NULL"
        assert _is_null("is_deleted", 0) == "is_deleted IS NOT NULL"

    # def test_is_null_type_error(self):
    #     with pytest.raises(TypeError):
    #         _is_null("is_deleted", "True")
    #         _is_null("is_deleted", "False")
    #         _is_null("is_deleted", "1")
    #         _is_null("is_deleted", "0")


class TestGroupBy:
    def test_group_by_string(self):
        assert _group_by("id") == "GROUP BY id"

    def test_group_by_list(self):
        assert _group_by(["id", "name", "age"]) == "GROUP BY id,name,age"

    def test_group_by_tuple(self):
        assert _group_by(("id", "name", "age")) == "GROUP BY id,name,age"

    # def test_group_by_invalid_type(self):
    #     with pytest.raises(TypeError):
    #         _group_by(123)
    #         _group_by({"id", "name", "age"})
    #         _group_by(("id", "name", "age"), "extra argument")


class TestOrderBy:
    def test_single_string_value(self):
        result = _order_by("created_at")
        assert result == "ORDER BY created_at DESC"

    def test_single_dict_value(self):
        result = _order_by({"created_at": True, "deleted_at": "desc"})
        assert result == "ORDER BY created_at DESC,deleted_at DESC"

    def test_multiple_list_values(self):
        result = _order_by([("created_at", True), ("deleted_at", "desc")])
        assert result == "ORDER BY created_at DESC,deleted_at DESC"

    # def test_invalid_type(self):
    #     with pytest.raises(ValueError):
    #         _order_by(123)

    def test_boolean_values(self):
        result = _order_by([("created_at", True), ("deleted_at", False)])
        assert result == "ORDER BY created_at ASC,deleted_at DESC"

    def test_enum_values(self):
        result = _order_by([{"created_at": OrderEnum.DESC.value, "deleted_at": OrderEnum.ASC.value}])
        assert result == "ORDER BY created_at DESC,deleted_at ASC"


class TestLimit:
    def test_single_int_value(self):
        result = _limit(1)
        assert result == "LIMIT 1"

    def test_multiple_int_values(self):
        result = _limit([1, 2])
        assert result == "LIMIT 1,2"

    def test_tuple_int_values(self):
        result = _limit((1, 2))
        assert result == "LIMIT 1,2"
    #
    # def test_invalid_type(self):
    #     with pytest.raises(TypeError):
    #         _limit("not an int or tuple/list of ints")


class TestOffset:
    def test_valid_value(self):
        result = _offset(10)
        assert result == "OFFSET 10"

    # def test_invalid_type(self):
    #     with pytest.raises(TypeError):
    #         _offset("not an int")


class TestEND:
    def test_int_value(self):
        # 测试整数类型
        assert _end("age", 18) == "age=18"
        assert _end("age", [18, 19, 20]) == "age IN (18,19,20)"
        assert _end("age", (18, 19, 20)) == "age IN (18,19,20)"

    def test_str_value(self):
        # 测试字符串类型
        assert _end("name", "Tom") == "name='Tom'"
        assert _end("name", ["Tom", "Jerry", "Spike"]) == "name IN ('Tom','Jerry','Spike')"
        assert _end("name", ("Tom", "Jerry", "Spike")) == "name IN ('Tom','Jerry','Spike')"

    def test_mix_value(self):
        # 测试混合类型
        assert _end("info", [1, "Tom", 3]) == "info IN (1,'Tom',3)"
        assert _end("info", (1, "Tom", 3)) == "info IN (1,'Tom',3)"

    def test_invalid_type(self):
        # 测试类型
        assert _end("age", "18") == "age='18'"
        assert _end("name", 18) == "name=18"
        assert _end("name", [18, "Tom"]) == "name IN (18,'Tom')"
        # assert _end("info", {"id": 1, "name": "Tom"}) == "info={'id': 1, 'name': 'Tom'}"


class TestOR:
    def test_empty(self):
        assert _or([]) == ""

    def test_single_condition(self):
        assert _or([("__end", "age=1")]) == "age=1"

    def test_multiple_conditions(self):
        assert _or([
            ("__end", "age=1"),
            ("__is_null", "created_at IS NOT NULL")
        ]) == "(age=1 OR created_at IS NOT NULL)"

    def test_multiple_conditions_with_limit(self):
        assert _or([
            ("__end", "age=1"),
            ("__is_null", "created_at IS NOT NULL"),
            ("__limit", "LIMIT 10")
        ]) == "(age=1 OR created_at IS NOT NULL LIMIT 10)"


class TestAnd:
    def test_empty_value(self):
        assert _and(value=[]) == ""

    def test_single_value(self):
        assert _and(value=[("__end", "age=1")]) == "age=1"

    def test_multiple_values(self):
        assert _and(value=[
            ("__end", "age=1"),
            ("__is_null", "created_at IS NOT NULL"),
            ("__order_by", "ORDER BY created_at DESC"),
            ("__extra", "(select * from user)"),
        ]) == "(age=1 AND created_at IS NOT NULL ORDER BY created_at DESC (select * from user))"

    def test_different_separator(self):
        assert _and(value=[
            ("__end", "age=1"),
            ("__is_null", "created_at IS NOT NULL"),
            ("__group_by", "GROUP BY age"),
        ]) == "(age=1 AND created_at IS NOT NULL GROUP BY age)"

    def test_kwargs(self):
        assert _and(value=[], foo="bar") == ""


class TestExtra:
    def test_str_extra(self):
        value = "on duplicate key update id=value(id)"
        assert _extra(value) == value

    def test_other_extra(self):
        value = {"name": 111}
        assert _extra(value) == str(value)


class TestMetaConditionMap:
    # Test suffix "__end"
    def test_suffix_end(self):
        result = meta_condition_map("name", 1)
        assert result == (_end, '__end', 'name', 1)

    # Test suffix "__gt"
    def test_suffix_gt(self):
        result = meta_condition_map("age__gt", 100)
        assert result == (_gt, "__gt", "age", 100)

    # Test suffix "__lt"
    def test_suffix_lt(self):
        result = meta_condition_map("age__lt", 100)
        assert result == (_lt, "__lt", "age", 100)

    # Test suffix "__e"
    def test_suffix_e(self):
        result = meta_condition_map("age__e", 100)
        assert result == (_e, "__e", "age", 100)

    # Test suffix "__gte"
    def test_suffix_gte(self):
        result = meta_condition_map("age__gte", 100)
        assert result == (_gte, "__gte", "age", 100)

    # Test suffix "__lte"
    def test_suffix_lte(self):
        result = meta_condition_map("age__lte", 100)
        assert result == (_lte, "__lte", "age", 100)

    # Test suffix "__ne"
    def test_suffix_ne(self):
        result = meta_condition_map("age__ne", 100)
        assert result == (_ne, "__ne", "age", 100)

    # Test suffix "__ll"
    def test_suffix_ll(self):
        result = meta_condition_map("name__ll", "ttt")
        assert result == (_ll, "__ll", "name", "ttt")

    # Test suffix "__rl"
    def test_suffix_rl(self):
        result = meta_condition_map("name__rl", "ttt")
        assert result == (_rl, "__rl", "name", "ttt")

    # Test suffix "__al"
    def test_suffix_al(self):
        result = meta_condition_map("name__al", 100)
        assert result == (_al, "__al", "name", 100)

    # Test suffix "__in"
    def test_suffix_in(self):
        result = meta_condition_map("user.id__in", [1, 2, 3])
        assert result == (_in, "__in", "user.id", [1, 2, 3])

    # Test suffix "__not_in"
    def test_suffix_not_in(self):
        result = meta_condition_map("id__not_in", [1, 2, 3])
        assert result == (_not_in, "__not_in", "id", [1, 2, 3])

    # Test prefix "__extra"
    def test_prefix_extra(self):
        result = meta_condition_map("__extra", {"key": "value"})
        assert result == (_extra, "__extra", None, {"key": "value"})

    # Test prefix "__limit"
    def test_prefix_limit(self):
        result = meta_condition_map("__limit", 10)
        assert result == (_limit, "__limit", None, 10)

    # Test prefix "__offset"
    def test_prefix_offset(self):
        result = meta_condition_map("__offset", 10)
        assert result == (_offset, "__offset", None, 10)

    # Test prefix "__group_by"
    def test_prefix_group_by(self):
        result = meta_condition_map("__group_by", "user.id")
        assert result == (_group_by, "__group_by", None, "user.id")

    # Test prefix "__order_by"
    def test_prefix_order_by(self):
        result = meta_condition_map("__order_by", {"created_at": True})
        assert result == (_order_by, "__order_by", None, {"created_at": True})

    # Test prefix "__or"
    def test_prefix_or(self):
        result = meta_condition_map("__or", ["age>10", "GROUP BY id"])
        assert result == (_or, "__or", None, ["age>10", "GROUP BY id"])

    # Test prefix "__and"
    def test_prefix_and(self):
        result = meta_condition_map("__and", ["age>10", "GROUP BY id"])
        assert result == (_and, "__and", None, ["age>10", "GROUP BY id"])

    # def test_invalid_flag(self):
    #     with pytest.raises(KeyError):
    #         meta_condition_map("__invalid_flag", "value")
    #
    # def test_invalid_suffix(self):
    #     with pytest.raises(KeyError):
    #         meta_condition_map("name__invalid_suffix", "value")
    #
    # def test_valid_suffix_without_prefix(self):
    #     with pytest.raises(ValueError):
    #         meta_condition_map("__order_by", "value")
    #
    # def test_valid_prefix_without_suffix(self):
    #     with pytest.raises(KeyError):
    #         meta_condition_map("__gt", "value", prefix="prefix")

    def test_valid_prefix_and_suffix(self):
        condition_fn, flag, prefix, value = meta_condition_map("name__gt", 100)
        assert callable(condition_fn)
        assert flag == "__gt"
        assert prefix == "name"
        assert value == 100

    def test_kwargs(self):
        condition_fn, flag, prefix, value = meta_condition_map("__order_by", {"created_at": True}, extra="extra")
        assert callable(condition_fn)
        assert flag == "__order_by"
        assert prefix is None
        assert value == {"created_at": True}

class TestGetPrefixAndSuffix:

    def test_normal_flag(self):
        assert get_prefix_and_suffix("name") == ("name", "__end")

    def test_flag_with_suffix(self):
        assert get_prefix_and_suffix("age__gt") == ("age", "__gt")

    def test_empty_flag(self):
        assert get_prefix_and_suffix("") == ("", "__end")

    def test_flag_with_double_underscores(self):
        assert get_prefix_and_suffix("field__with__double__underscores") == ("field__with__double", "__underscores")

    def test_flag_with_underscores_only(self):
        assert get_prefix_and_suffix("__") == ("__", "__end")

    def test_flag_with_prefix_only(self):
        assert get_prefix_and_suffix("prefix__") == ("prefix__", "__end")

    def test_flag_with_numeric_suffix(self):
        assert get_prefix_and_suffix("number__123") == ("number", "__123")

    def test_flag_with_prefix_and_numeric_suffix(self):
        assert get_prefix_and_suffix("name__456") == ("name", "__456")

    def test_flag_empty_prefix(self):
        assert get_prefix_and_suffix("__456") == ("__456", "__end")

    # def test_get_prefix_and_suffix_none(self):
    #     assert get_prefix_and_suffix(None) == (None, "__end")


class TestGeneralProcessLayer:

    def test_sql_injection_prevention(self):
        query_params = []
        flag, value = ("user.id__in", [1, 2, 3])
        _, result = general_process_layer(flag, value, query_params=query_params)

        assert result == "user.id IN (%s,%s,%s)"
        assert query_params == [1, 2, 3]

    def test_single_value(self):
        query_params = []
        flag, value = ("age__gt", 100)
        _, result = general_process_layer(flag, value, query_params=query_params)

        assert result == "age>%s"
        assert query_params == [100]

    def test_multiple_values(self):
        query_params = []
        flag, value = ("user.id__in", [1, 2, 3])
        _, result = general_process_layer(flag, value, query_params=query_params)

        assert result == "user.id IN (%s,%s,%s)"
        assert query_params == [1, 2, 3]

    def test_order_by(self):
        query_params = []
        flag, value = "__order_by", {"created_at": True}
        _, result = general_process_layer(flag, value, query_params=query_params)

        assert result == "ORDER BY created_at DESC"
        assert query_params == []

    def test_offset(self):
        query_params = []
        flag, value = "__offset", 10
        _, result = general_process_layer(flag, value, query_params=query_params)

        assert result == "OFFSET %s"
        assert query_params == [10]


class TestHookFunc:
    def test_default_hook(self):
        # 测试默认空钩子，返回的结果应该是参数本身
        meta_cond = MetaCondition(flag="__limit", value=10)
        kwargs_params = {"hooks": [], "hook_error": False}
        _, kwargs = HookFunc.execute(meta_cond, **kwargs_params)
        assert kwargs == kwargs_params

    def test_sql_preprocess_hook_single_value(self):
        # 测试SQL预处理钩子单值情况，返回的结果应该是替换%s并将参数加入query_params列表
        meta_cond = MetaCondition(flag="__gt", value=10, flag_sign="__gt", field="age")
        _, kwargs = HookFunc.execute(meta_cond)

        assert kwargs == {"hooks": ["sql_preprocess_hook"], "query_params": [10], "strict_mode": False}
        assert meta_cond.flag_value == "%s"

    def test_sql_preprocess_hook_multi_value(self):
        # 测试SQL预处理钩子多值情况，返回的结果应该是替换%s并将参数加入query_params列表
        meta_cond = MetaCondition(flag="__in", value=[1, 2, 3], flag_sign="__in", field="id")
        _, kwargs = HookFunc.execute(meta_cond)

        assert kwargs == {"query_params": [1, 2, 3], "strict_mode": False, "hooks": ["sql_preprocess_hook"]}
        assert meta_cond.flag_value == ["%s", "%s", "%s"]

    def test_execute_with_user_hook(self):
        # 测试自定义钩子，返回的结果应该是执行自定义钩子后的结果
        class CustomHook(HookFunc):
            @staticmethod
            def custom_hook(meta_cond: MetaCondition, **kwargs):
                meta_cond.flag_sign = "__custom_hook"
                kwargs["custom_param"] = True
                return meta_cond, kwargs

        meta_cond = MetaCondition(flag="__eq", value=10, flag_sign="__eq", field="age")
        _, kwargs = CustomHook.execute(meta_cond, use_hook=True, hooks=["custom_hook"])
        assert kwargs == {"custom_param": True, "use_hook": True, "hooks": ["custom_hook"]}
        assert meta_cond.flag_sign == "__custom_hook"

    def test_execute_without_user_hook(self):
        # 测试关闭钩子，返回的结果应该是参数本身
        meta_cond = MetaCondition(flag="__gt", value=10, flag_sign="__gt", field="age")
        _, kwargs = HookFunc.execute(meta_cond, use_hook=False)
        assert kwargs == {"use_hook": False}


class TestConditionParse:
    def test_simple_equal(self):
        value = {"flag": "__and", "value": {"name": "ttt"}, "use_hook": False}
        expected_result = ("__and", "name='ttt'")
        assert condition_parse(**value) == expected_result

    def test_simple_equal_and_use_hook(self):
        value = {"flag": "__and", "value": {"name": "ttt"}}
        expected_result = ("__and", "name=%s")
        assert condition_parse(**value) == expected_result

    def test_in(self):
        value = {"flag": "__and", "value": {"age__in": [1, 2]}}
        expected_result = ("__and", "age IN (%s,%s)")
        assert condition_parse(**value) == expected_result

    def test_lt_and_gt(self):
        value = {"flag": "__and", "value": {"age__lt": 10, "height__gt": 150}}
        expected_result = ("__and", "(age<%s AND height>%s)")
        assert condition_parse(**value) == expected_result

    def test_extra(self):
        value = {"flag": "__and", "value": {"__extra": "group by name,id"}}
        expected_result = ("__and", "group by name,id")
        assert condition_parse(**value) == expected_result

    def test_invalid_operation(self):
        value = {"flag": "__and", "value": {"age__lt": None, "height__gt": 150}}
        expected_result = ("__and", "")
        assert condition_parse(**value) == expected_result

    def test_nested_dict(self):
        value = {
            "flag": "__and",
            "value": {
                "name": "ttt",
                "age__in": [1, 2],
                "__or": {
                    "user.id": [1, 2, 3],
                    "__or": {
                        "1__extra": "(一段自定义SQL)",
                        "age__lt": 10,
                        "name": "jjj",
                        "is_deleted__is_null": False,
                        "created_at": None,
                        "2__extra": "(另一段自定义SQL)",
                        "__extra": "(最后一段自定义SQL)",
                    }
                }
            }
        }
        expected_result = ("__and", "(name=%s AND age IN (%s,%s) AND (user.id IN (%s,%s,%s) OR ((一段自定义SQL) OR age<%s OR name=%s OR is_deleted IS NOT NULL OR (另一段自定义SQL) OR (最后一段自定义SQL))))")
        assert condition_parse(**value) == expected_result


def test_db_condition_parse():
    # 测试空条件
    # with pytest.raises(TypeError):
    #     db_condition_parse({})

    # 测试全等(=)、不等(!=)、大于(>)、大于等于(>=)、小于(<)、小于等于(<=)、左模糊匹配(LIKE)、右模糊匹配(LIKE)、全模糊匹配(LIKE)、
    # 在列表(IN)、不在列表(NOT IN)、空值判断(IS NULL, IS NOT NULL)、分页限制(LIMIT)、位移(OFFSET)、排序(ORDER BY)、分组(GROUP BY)、
    # 与操作(AND)、或操作(OR)、自定义操作(eg: 支持在任意位置添加任意数量的自定义操作)
    cond = {
        "name": "Alice",
        "age__gt": 18,
        "created_at__gte": "2022-01-01",
        "updated_at__lt": "2022-02-01",
        "email__ll": "%example.com",
        "status__in": [1, 2, 3],
        "status__not_in": [4, 5, 6],
        "address__is_null": True,
        "__limit": 10,
        "__offset": 20,
        "__order_by": [("created_at", True), ("updated_at", False)],
        "__group_by": ["age"],
        "__and": {"gender": "female"},
        "__or": {"age__lt": 10, "sex": 1},
        "__extra": "some extra conditions"
    }
    expected_query_params = ['Alice', '2022-01-01', '2022-02-01', '%example.com', 1, 2, 3, 4, 5, 6]
    expected_query_str = (
        "name=%s AND age>%s AND created_at>=%s AND updated_at<%s AND email LIKE %%s AND status IN (%s,%s,%s) "
        "AND status NOT IN (%s,%s,%s) AND address IS NULL LIMIT 10 OFFSET 20 ORDER BY created_at ASC, updated_at DESC "
        "GROUP BY age AND gender=%s AND (age<%s OR sex=%s) some extra conditions"
    )
    query_str, query_params = db_condition_parse(cond)
    # assert query_str == expected_query_str
    # assert query_params == expected_query_params

    # # 测试空值判断
    # cond = {"name": None, "age": False, "gender": []}
    # expected_query_params = []
    # expected_query_str = ""
    # query_str, query_params = db_condition_parse(cond)
    # assert query_str == expected_query_str
    # assert query_params == expected_query_params
