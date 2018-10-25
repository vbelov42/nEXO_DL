#!/usr/bin/env python
# coding: utf-8

# In[7]:


import csv

with open('image2dcharge.csv', 'w') as csvfile:
    fieldnames = ['filename', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #writer.writeheader()
    for i in range(39000):
        writer.writerow({'filename': './images/gamma%d.jpg' % i, 'label': '0'})
    for j in range(39000):
        writer.writerow({'filename': './images/electron%d.jpg' % i, 'label': '1'})


# In[ ]:




