import os
import csv

NR_FRAMES_TO_SEC = {32: 1, 94: 3, 157: 5, 188: 6, 313: 10, 628: 20, 911: 30}
csvs_dir = r'...'

files = [(csvs_dir + os.sep + file, file) for file in os.listdir(csvs_dir)]
results = {'song': [], 'frame': []}

for file in files:
    with open(file[0], mode='r') as csv_file:
        f1_scores = [float(row[3]) for row in csv.reader(csv_file) if row[3] != 'f1-score']
        max_f1_score = round(max(f1_scores), 4)
        mean_f1_score = round(sum(f1_scores) / len(f1_scores), 4)

        information = file[1].split('_')
        feature_type = 'song' if information[1] == 'pooled' else 'frame'
        seconds = NR_FRAMES_TO_SEC[int(information[0])]

        results[feature_type].append((seconds, mean_f1_score, max_f1_score))

results['song'] = sorted(results['song'], key=lambda x: x[0])
results['frame'] = sorted(results['frame'], key=lambda x: x[0])

print('song')
print('seconds | mean | max')
for seconds, mean_f1, max_f1 in results['song']:
    print(f'{seconds} | {mean_f1} | {max_f1}')
print()

print('frame')
print('seconds | mean | max')
for seconds, mean_f1, max_f1 in results['frame']:
    print(f'{seconds} | {mean_f1} | {max_f1}')
print()
