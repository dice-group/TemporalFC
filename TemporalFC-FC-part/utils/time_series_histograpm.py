import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate some random data
def load_data(data_dir, data_type, pred=False):
    try:
        data = []
        year_data = []
        if pred == False:
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                for datapoint in f:
                    datapoint = datapoint.split()
                    if len(datapoint) == 4:
                        s, p, o, label = datapoint
                        if label == 'True':
                            label = 1
                        else:
                            label = 0
                        data.append((s, p, o, label))
                    elif len(datapoint) == 3:
                        s, p, label = datapoint
                        assert label == 'True' or label == 'False'
                        if label == 'True':
                            label = 1
                        else:
                            label = 0
                        data.append((s, p, 'DUMMY', label))
                    elif len(datapoint) >= 5:
                        s, p, o, year, label = datapoint[0:5]

                        if label == 'True':
                            label = 1
                        else:
                            label = 0
                        year = year.replace(".0", "").replace('<', '').replace('>', '')
                        if int(year) >= 1900 and int(year) <= 2022:
                            year_data.append(int(year))
                            data.append((s, p, o, "<" + year.replace(".0", "") + ">", label))
                    else:
                        raise ValueError
        else:
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                for datapoint in f:
                    datapoint = datapoint.split()
                    if len(datapoint) == 5:
                        s, p, o, label, dot = datapoint
                        data.append((s, p, o, label.replace("\"^^<http://www.w3.org/2001/XMLSchema#double>", "")))
                    elif len(datapoint) == 4:
                        s, p, o, label = datapoint
                        data.append((s, p, o, label))
                    elif len(datapoint) == 3:
                        s, p, label = datapoint
                        data.append((s, p, 'DUMMY', label))
                    else:
                        raise ValueError
    except FileNotFoundError as e:
        print(e)
        print('Add empty.')
        data = []
    except:
        print("test")
    return year_data, data


year, train_set = list((load_data("../dataset/Yago3K/orignal_data/", data_type="train_original")))

year_count = []
for i in range(1900, 2022):
    year_count.append(year.count(i))

x = np.arange(1900,2022)
y = np.arange(1900,2022)
z = year_count
jj = 1900
with open("../dataset/Yago3K/orignal_data/"+'year_count.txt', 'w') as f:
    for i in range(len(year_count)):
        f.write(f'{jj} {year_count[i]}\n')
        jj = jj+1
# Create a 3D histogram using the hist3d() function
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# hist, xedges, yedges = np.histogram2d(x, y, bins=20)
# xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0
# dx = dy = 0.5 * np.ones_like(zpos)
# dz = hist.ravel()
# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
#
# # Add labels and titles
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_title('3D Histogram')
#
# # Show the plot
# plt.show()
#
#
# import numpy as np
# import matplotlib.pyplot as plt

# Create some example data
# x = np.random.normal(size=1000)
# y = np.random.randint(low=1900, high=2023, size=1000)

x = np.array(year_count)
y = np.array(y)

# Create a 2D histogram
# plt.hist2d(y, x, bins=10, cmap='Blues')
# create a figure and axis object
fig, ax = plt.subplots()

# plot the data points
ax.scatter(y, x)

# Set the y-axis limits to show the range of years
plt.xlim(1900, 2022)

# Add a colorbar
# cb = plt.colorbar()
# cb.set_label("Intensity", rotation=270)
# Add labels and a title
plt.xlabel('Years')
plt.ylabel('Count')
# plt.title('2D Histogram with Year Counts on X-Axis')

# Show the plot
plt.show()


#
# import matplotlib.pyplot as plt
#
# # create data points
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]
#
# # create a figure and axis object
# fig, ax = plt.subplots()
#
# # plot the data points
# ax.scatter(x, y)
#
# # set axis labels
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
#
# # set the title of the plot
# ax.set_title('Data Points')
#
# # display the plot
# plt.show()
