import json
import re
import urllib.parse
import torch


def string_to_ascii_tensor(string, max_len=100):
    ascii_codes = [0 if ord(c) - 32 < 0 or ord(c) - 32 > 95 else ord(c) - 32 for c in string]
    if len(ascii_codes) > max_len:
        ascii_codes = ascii_codes[:max_len]
    else:
        ascii_codes += [0] * (max_len - len(ascii_codes))
    return torch.tensor(ascii_codes, dtype=torch.long)


def store_actions_logs(actions_logs, file_path):
    with open(file_path, 'w') as f:
        json.dump(actions_logs, f)


# 两个字符串在去掉空格之后是否相等
def compare_strings_ignore_spaces(str1, str2):
    # 去除所有空格后比较字符串
    return str1.replace(" ", "") == str2.replace(" ", "")


"""
下面是action_space
"""


def skip(text):
    return text


# 将原始字符串的30=30替换为1=1
def replace_equals_to_true(text):
    # 定义正则表达式模式，匹配形如 "1=1", "2=2", "30=30" 等等表达式
    pattern = re.compile(r'\b(\d+)\s*=\s*\1\b')

    # 使用 sub 函数替换匹配的表达式为 "1=1"
    return pattern.sub('1=1', text)


# 删除字符串中最后一个#号之后的内容（保留#）
def remove_comments_2a(text):
    return text.rsplit("#", 1)[0] + "#"


# 删除字符串中最后一个--号之后的内容（保留--）
def remove_comments_2b(text):
    return text.rsplit("--", 1)[0] + "--"


def remove_comments(string):
    # 正则表达式匹配普通注释，但不匹配以 /*! 开头的注释
    pattern = r'/\*(?!\!)(.*?)\*/'
    cleaned_string = re.sub(pattern, ' ', string, flags=re.DOTALL)
    return cleaned_string


def remove_comments_3(string):
    # 将 /*! 和 */ 去掉，保留其中的内容
    special_pattern = r'/\*\!(.*?)\*/'
    cleaned_string = re.sub(special_pattern, r'\1', string, flags=re.DOTALL)
    return cleaned_string


def DML_substitution(input_string):
    # 定义需要替换的模式和对应的替换字符串
    replacements = {
        r'\s*like\s*': '=',  # 将\s*like\s*替换为=
        r'&&': 'AND',  # 将&&替换为 and
        r'\|\|': 'OR',  # 将||替换为 or
        # 可根据需要继续添加其他替换规则
    }

    # 遍历替换字典中的每个模式和替换字符串，并使用re.sub()进行替换
    for pattern, replacement in replacements.items():
        input_string = re.sub(pattern, replacement, input_string)

    return input_string


# 把一些关键词转换为大写形式
def toggle_case(input_string):
    target_words = ['SELECT', 'UNION', 'OR', 'AND']
    # 定义所有分隔符
    delimiters = r'[!#$%^&*(){}@~;, =<>\[\]_?|]+'

    # 使用正则表达式分割字符串，并保留分隔符
    parts = re.split(f'({delimiters})', input_string)

    # 遍历每个部分，如果是目标词汇的大小写组合，则替换为大写
    for i in range(len(parts)):
        if parts[i].upper() in target_words:
            parts[i] = parts[i].upper()

    # 重新将分割的部分组合成完整字符串
    return ''.join(parts)


def hex2decimal(input_string):
    # return input_string
    # 定义匹配十六进制数字的正则表达式模式
    pattern = r'\b0[xX][0-9a-fA-F]+\b'

    # 使用 re.sub() 并通过一个替换函数进行转换
    def replace_hex(match):
        hex_number = match.group(0)  # 获取匹配到的十六进制数字
        decimal_number = str(int(hex_number, 16))  # 将其转换为十进制数字
        return decimal_number  # 返回替换后的十进制数字

    # 使用 re.sub() 进行替换
    result_string = re.sub(pattern, replace_hex, input_string)

    return result_string


def placeholders_replace(string):
    pattern = r'\\t|\\n'
    result = re.sub(pattern, ' ', string)
    return result


def decode_url_encoding(input_string):
    # 定义匹配URL编码的正则表达式模式
    pattern = r'%[0-9a-fA-F]{2}'

    # 使用re.sub()和一个替换函数进行解码
    def replace_encoded(match):
        encoded_str = match.group(0)  # 获取匹配到的URL编码字符串
        decoded_str = urllib.parse.unquote(encoded_str)  # 解码URL编码字符串
        return decoded_str  # 返回解码后的字符串

    # 使用re.sub()进行替换
    result_string = re.sub(pattern, replace_encoded, input_string)

    return result_string


def evaluate_simple_expression(expr):
    expr2 = expr.replace('=', '==')
    expr2 = expr2.replace('<>', '!=')
    try:
        # 尝试评估表达式，如果失败则返回原始表达式
        return eval(expr2)
    except:
        return expr


# 简化布尔表达式，暂时不用
def simplify_logical_expressions(text):
    logical_operators = r'\band\b|\bor\b|\bnot\b|&&|\|\|'

    # 定义模式来匹配逻辑表达式
    pattern = rf'(\S+\s*({logical_operators})\s*\S+)'

    def simplify_once(text):
        matches = re.findall(pattern, text)
        if not matches:
            return text

        for match in matches:
            expr = match[0]
            left, operator, right = re.split(r'\s*({})\s*'.format(logical_operators), expr, maxsplit=1)

            left_eval = evaluate_simple_expression(left)
            right_eval = evaluate_simple_expression(right)

            simplified_expr = expr  # 默认不简化表达式

            if operator == 'or':
                if left_eval == True or left_eval == 'true':
                    simplified_expr = 'true'
                elif right_eval == True or right_eval == 'true':
                    simplified_expr = 'true'
                elif left_eval == False or left_eval == 'false':
                    simplified_expr = right
                elif right_eval == False or right_eval == 'false':
                    simplified_expr = left
            elif operator == 'and':
                if left_eval == False or left_eval == 'false':
                    simplified_expr = 'false'
                elif right_eval == False or right_eval == 'false':
                    simplified_expr = 'false'
                elif left_eval == True or left_eval == 'true':
                    simplified_expr = right
                elif right_eval == True or right_eval == 'true':
                    simplified_expr = left

            text = text.replace(expr, str(simplified_expr), 1)

        return text

    # 连续替换直至无变化
    # previous_text = None
    # while text != previous_text:
    #     previous_text = text
    #     text = simplify_once(text)
    #     text = re.sub(r'\s+', ' ', text).strip()

    # 只替换一次
    text = simplify_once(text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def convert_harmless_subquery_to_true(sql_query):
    # 正则表达式模式，匹配形如 (SELECT 数字) 的子查询
    pattern = re.compile(r'\(select\s+\d+\s*\)', re.IGNORECASE)

    # 将匹配的子查询替换为 True
    converted_query = pattern.sub('1=1', sql_query)

    return converted_query
