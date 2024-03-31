space_list = [1, 2, 3, 4, 5, 6]
variables = ['CO2', 'NO', 'Ozone', 'PM10', 'PM2.5']

result_list = [f"{i} : {v}" for i in space_list for v in variables]

print(result_list)

for s in (result_list):
    print(s)