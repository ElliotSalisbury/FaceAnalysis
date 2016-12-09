import os
import csv

RateMeFolder = "E:\\Facedata\\RateMe"

submissionFolders = [x[0] for x in os.walk(RateMeFolder)]

#compile into single file
with open(os.path.join(RateMeFolder, "combined.csv"), 'w', newline='') as wf:
    writer = csv.writer(wf)
    writer.writerow(("Submission Title", "Submission Age", "Submission Gender", "Submission Author", "Rating Author","Rating", "Decimal", "Rating Text"))

    for folder in submissionFolders:
        ratingsPath = os.path.join(folder, "ratings.csv")

        if not os.path.exists(ratingsPath):
            continue

        with open(ratingsPath, 'r') as rf:
            reader = csv.reader(rf)

            for i, row in enumerate(reader):
                if i==0:
                    continue
                writer.writerow(row)