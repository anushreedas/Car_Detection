import csv
import math

ALPHA = 4


def get_csv_data(file_name):
    data = []
    with open(file_name, newline='') as csvfile:
        detections = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in detections:
            data.append((row[0], row[1], row[2], row[3], row[4]))

    return data

# for the normal distribution
def normal_distribution(time):
    return 1/(2*math.sqrt(2*math.pi)) * math.exp(-1/2 * ((time - 12)/2)**2)


def unbounded_equation(time, detected_cars, weather, lot_capacity):
    if detected_cars < 3:
        return 100
    else:
        return 1/(detected_cars * weather * ALPHA * normal_distribution(time))


def predict_equation(time, detected_cars, weather, lot_capacity):
    p = unbounded_equation(time, detected_cars, weather, lot_capacity)
    return p/(p+ALPHA)

if __name__ == "__main__":
    data = get_csv_data('detections.csv')
    predictions = []
    for dat in data:
        predictions.append(predict_equation(int(dat[2]), int(dat[1]), float(dat[3]), int(dat[4])))

    with open('predictions.csv', 'w+') as predic:
        for prediction in predictions:
            predic.write(str(prediction) + '\n')

