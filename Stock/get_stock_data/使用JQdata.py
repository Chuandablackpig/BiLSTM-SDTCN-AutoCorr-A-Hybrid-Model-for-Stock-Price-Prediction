from jqdatasdk import *
auth(username= '13385059120', password= '1980368268aA')

# 查询当日剩余可调用条数，每日100万条
get_query_count()
infos = get_account_info()
print(infos)

