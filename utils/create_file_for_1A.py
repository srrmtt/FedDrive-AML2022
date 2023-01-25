import random

def create_file(fileI1,fileI2,fileO1,fileO2):
  images = []
  train_images = []
  test_images = []
  with open(fileI1) as f1:
    images = f1.readlines()
    with open(fileI2) as f2:
      images += f2.readlines()
      images = sorted(images)
      city = images[0].split('/')[0]
      same_city = []
      for img in images:
        curr_city = img.split('/')[0]
        if city != curr_city:
          for i in range(2):
            num = 0 if len(same_city) == 1 else random.randint(0,len(same_city)-1)
            e = same_city.pop(num)
            test_images.append(e)
          train_images += same_city
          same_city = []
          city = curr_city
        same_city.append(img)
      for i in range(2):
        num = 0 if len(same_city) == 1 else random.randint(0,len(same_city)-1)
        e = same_city.pop(num)
        test_images.append(e)
      train_images += same_city
      print(len(train_images))
      print(len(test_images))
      out_train = open(fileO1,'w')
      out_train.writelines(train_images)
      out_train.close()
      out_train = open(fileO2,'w')
      out_train.writelines(test_images)
      out_train.close()