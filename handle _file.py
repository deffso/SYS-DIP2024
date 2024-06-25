import zipfile

f=zipfile.ZipFile('train.zip','r')
for file in f.namelist():
    print(file)
    f.extract(file,"")
f.close()

f=zipfile.ZipFile('image.zip','r')
for file in f.namelist():
    print(file)
    f.extract(file,"test/")
f.close()
