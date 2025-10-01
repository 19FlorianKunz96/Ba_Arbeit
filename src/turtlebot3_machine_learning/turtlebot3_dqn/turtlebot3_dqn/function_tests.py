import uuid
import datetime
import os 

uuid1= uuid.uuid4()
today=datetime.date.today()
print(f'{uuid1}_{today}')

with open(os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'trainings_done'),'7a59a99f-844c-4f45-ae06-7fc0e1f33e29_2025-10-01_stage3'), 'test.txt'), 'w') as f:
    f.write('hello')