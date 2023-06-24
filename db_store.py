import project_module as pm

if "__name__" == "__main__":
    df = pm.datagovsg_api_call('https://data.gov.sg/api/action/datastore_search?resource_id=f1765b54-a209-4718-8d38-a39237f502b3', 
                            sort='month desc',
                            limit = 1000000,
                            months = [1,2,3,4,5,6,7,8,9,10,11,12],
                            years=[2020,2021,2022,2023])
    
    