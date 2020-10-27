''' Usage :
    python test_train_split.py -d ~/mask-detection/annotations/
'''
import glob, os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="Directory of dataset")
args = vars(ap.parse_args())


img_dir = str(args["dir"])
print(args["dir"])
percentage_test = 10;

file_test = open(r'C:\Users\CHEON\Desktop\mteg\mask-detection\test.txt', 'w')

for pathAndFilename in glob.iglob(os.path.join(img_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_test = open(r'C:\Users\CHEON\Desktop\mteg\mask-detection/train_test/test.txt', 'a')
    file_test.write(img_dir + "/" + title + '.jpg' + "\n")
    file_test.close()