import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
from google.colab.patches import cv2_imshow

## TO DO: parse these parameters as arguments with argparse ##
No_of_datasets = 1000 # No of total datasets generated, also number of images that will be generated
total_samples = 30 # No of points in each image
train_test_ratio = 0.7 # 70% data will be used to train, 20% to test
class_names = ["points"] # by default only points are generated
linear_data_ratio = 1.0

def gen_data(total_samples, No_of_datasets):
  distribution_param = {}
  data = []
  dist_type = []
  for i in range(No_of_datasets):
    rnd = random.randint(1,8)
    if rnd == 1:
      a,b = 1,1000
      data.append([np.random.uniform(1,1000,total_samples)])
      dist_type.append("uniform")
      distribution_param["uniform"] = ("a = "+str(a)+", b = "+ str(b))
    elif rnd == 2:
      n,p = 10,0.5
      data.append([np.random.binomial(n,p,total_samples)])
      dist_type.append("binomial")
      distribution_param["binomial"] = ("n = "+str(n)+", p = "+ str(p))
    elif rnd == 3:
      #chi2 takes df as a shape parameter.
      df = 2
      data.append([np.random.chisquare(df,total_samples)])
      dist_type.append("chisquare")
      distribution_param["chisquare"] = ("df = " + str(df))

    elif rnd == 4:
      lam = 2
      data.append([np.random.exponential(1/lam,total_samples)])
      dist_type.append("exponential")
      distribution_param["exponential"] = ("lambda = "+ str(lam))

    elif rnd == 5:
      shape,scale = 2,2
      data.append([np.random.gamma(shape, scale, total_samples)])
      dist_type.append("gamma")
      distribution_param["gamma"] = ("shape = "+str(shape)+", scale = "+ str(scale))

    elif rnd == 6:
      mu, sigma = 0, 0.1 # mean and standard deviation
      data.append([np.random.normal(mu, sigma, total_samples)])
      dist_type.append("normal")
      distribution_param["normal"] = ("mu = "+str(mu)+", sigma = "+ str(sigma))

    elif rnd == 7:
      lam = 5
      data.append([np.random.poisson(lam, total_samples)])
      dist_type.append("poisson")
      distribution_param["poisson"] = ("lambda = "+ str(lam))

    else:
      a = 5. # shape
      data.append([np.random.power(a, total_samples)])
      dist_type.append("power")
      distribution_param["power"] = ("shape = "+ str(a))

  return data,dist_type,distribution_param


# Generate Datasets with total_samples number of points
x_data,x_dist_type,x_distribution_param = gen_data(total_samples,No_of_datasets)

y_data = []
y_dist_type = []
y_distribution_param = {}
number_of_linear_datasets = int(No_of_datasets*linear_data_ratio)
y_data = x_data[:number_of_linear_datasets]
y_dist_type = x_dist_type[:number_of_linear_datasets]
y_distribution_param = {dist: x_distribution_param[dist] for dist in y_dist_type}   # because distribution params are fixed


y_data_d,y_dist_type_d,y_distribution_param_d = gen_data(total_samples,No_of_datasets-number_of_linear_datasets)
y_data += y_data_d
y_dist_type += y_dist_type_d
y_distribution_param.update(y_distribution_param_d)


data = []
for (x,y) in zip(x_data,y_data):
  data.append([np.ndarray.tolist(x[0]),np.ndarray.tolist(y[0])])

dataset = []
for i,d in enumerate(data):
  d = pd.DataFrame(np.transpose(d), columns=[x_dist_type[i]+'_1',y_dist_type[i]+'_2'])
  # d = pd.DataFrame(np.transpose(d), columns=[x_dist_type[i],y_dist_type[i]])
  dataset.append(d)
  d.to_csv(r'data/custom/dataset_csv/'+str(i+1)+'.csv')

def get_centre_from_bbox(bbox_arr,height):
  centre_coords = []
  for bbox_wrapper in bbox_arr:
    box_width = np.abs(bbox_wrapper.x0-bbox_wrapper.x1)
    box_height = np.abs(bbox_wrapper.y0-bbox_wrapper.y1)
    centre_x = bbox_wrapper.x0 + (box_width/2)
    centre_y = (height - bbox_wrapper.y0) - (box_height/2)
    centre_coords.append([centre_x,centre_y,box_width,box_height])  

  return centre_coords

def gen_scatterplot(dataset,x_dist_type,y_dist_type,i,x_distribution_param,y_distribution_param,total_samples,num_train):

  meta_data = {}
  #1. X and Y axes labels
  columns = dataset.columns
  col1 = columns[0]
  col2 = columns[1]
  meta_data["xlabel_ylabel"] = (col1,col2)

  #2. Legends: Not discovered yet

  #3. Image dimensions, padding
  x = random.randint(7,12)
  y = random.randint(7,12)
  x = 4.16
  y = 4.16
  figsize=(x,y)
  meta_data["image_dim"] = (x,y)

  #4. Opacity of points
  alpha = random.uniform(0.5,1)
  meta_data["opacity"]  = (alpha)

  #5. Markers used for the points
  markers = ['o','v','x','+']
  marker = markers[random.randint(0,3)]
  meta_data["marker"] = (marker)

  #7. Colors
  r,g,b = (random.random(),random.random(),random.random())
  colors = (r,g,b)
  meta_data["colors"] = colors

  #8. Font size and Marker size
  font_size = random.randint(10,20)
  plt.rcParams.update({'font.size': font_size})
  ms = random.randint(3,10)
  marker_size = np.pi*ms
  meta_data["font_size_marker_size"] = (font_size,marker_size)

  #9. Diversity - Underlying distribution
  distributions = (x_dist_type,y_dist_type)
  meta_data["x_distr_y_distr"] = distributions

  #10 and 11. Parameters of distribution
  meta_data["param_x_distr"] = x_distribution_param[x_dist_type]
  meta_data["param_y_distr"] = y_distribution_param[y_dist_type]

  #12. Total Number of Samples
  meta_data["no_of_samples"]  = (total_samples)

  fig,ax = plt.subplots(figsize = figsize)
  scattered = ax.scatter(dataset[col1], dataset[col2], s=marker_size, alpha=alpha,  c=np.array([colors]), marker = marker )
  plt.title('Scatter plot')
  plt.xlabel(col1)
  plt.ylabel(col2)
  #plt.legend()   #Removed legend

  #6. Scale of axes  
  xticks = list(ax.get_xticks())
  yticks = list(ax.get_yticks())
  x_lower_lim,x_upper_lim = xticks[1],xticks[len(xticks)-2]
  y_lower_lim,y_upper_lim = yticks[1],yticks[len(yticks)-2]
  meta_data["x_scale_y_scale"]  = ((x_lower_lim,x_upper_lim),(y_lower_lim,y_upper_lim))

  # plt.gca().add_patch(patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none'))
  # plt.show()

  meta_data["data_filename"] = (str(i+1)+'.csv')
  path = 'data/custom/images/'+str(i+1)+'.jpg'
  plt.savefig(r'data/custom/images/'+str(i+1)+'.jpg')

  # cv2_imshow(img_read)

  if i<=num_train:
    with open("data/custom/train.txt",'a') as train_list:
      if i==num_train:
        train_list.write(path)
      else:
        train_list.write(path+"\n")
  else:
    with open("data/custom/valid.txt",'a') as valid_list:
      if i==len(dataset)-1:
        valid_list.write(path)
      else:
        valid_list.write(path+"\n") 

  mdata = pd.DataFrame(meta_data.keys())
  mdata['1'] = (meta_data.values())
  mdata.to_csv(r'data/custom/dataset_metadata/'+str(i+1)+'.txt',index= False,header= False)


  xy_pixels = ax.transData.transform(np.vstack([dataset[col1],dataset[col2]]).T)
  xpix, ypix = xy_pixels.T

  # In matplotlib, 0,0 is the lower left corner, whereas it's usually the upper 
  # left for most image software, so we'll flip the y-coords...
  width, height = fig.canvas.get_width_height()
  ypix = height - ypix

  bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  width, height = bbox.width*fig.dpi, bbox.height*fig.dpi

  # print("Dimensions of image are:")
  # print(width,height)
  # print("Box size")
  bounding_box = fig.dpi*5/90.0

  x_tick_pos = [ ax.transLimits.transform(textobj.get_position()) for textobj in ax.get_xticklabels() if len(textobj.get_text())>0]
  y_tick_pos = [ ax.transLimits.transform(textobj.get_position()) for textobj in ax.get_yticklabels() if len(textobj.get_text())>0]

  x_tick_pos = [ ax.transScale.transform(ax.transAxes.transform([array[0], 0])) for array in x_tick_pos]    #  axes_x_pos = 0
  y_tick_pos = [ ax.transScale.transform(ax.transAxes.transform([0, array[1]])) for array in y_tick_pos]

  x_tick_pos = [[xtick[0],height  - xtick[1]] for xtick in x_tick_pos]
  y_tick_pos = [[ytick[0],height  - ytick[1]] for ytick in y_tick_pos]

  x_tick_pos = x_tick_pos[1:-1]
  y_tick_pos = y_tick_pos[1:-1]
  tick_box_size = fig.dpi*5/50.0
  
  x_label_bounds = [ textobj.get_window_extent() for textobj in ax.get_xticklabels() if len(textobj.get_text())>0]
  y_label_bounds = [ textobj.get_window_extent() for textobj in ax.get_yticklabels() if len(textobj.get_text())>0]

  x_label_coords = get_centre_from_bbox(x_label_bounds[1:-1],height)
  y_label_coords = get_centre_from_bbox(y_label_bounds[1:-1],height)

  
  # for item_x_label_bounds in x_label_coords:
  #   x_0 = int(item_x_label_bounds[0])
  #   width_x0 = int(item_x_label_bounds[2])
  #   y_0 = int(item_x_label_bounds[1])
  #   height_y0 = int(item_x_label_bounds[3])
  #   #print(x_0,y_0,width_x0,height_y0)
  #   cv2.rectangle(img_read,(x_0-width_x0,y_0-height_y0),(x_0+width_x0,y_0+height_y0),(0,255,0),2)

  # for item_x_label_bounds in y_label_coords:
  #   x_0 = int(item_x_label_bounds[0])
  #   width_x0 = int(item_x_label_bounds[2])
  #   y_0 = int(item_x_label_bounds[1])
  #   height_y0 = int(item_x_label_bounds[3])
  #   #print(x_0,y_0,width_x0,height_y0)
  #   cv2.rectangle(img_read,(x_0-width_x0,y_0-height_y0),(x_0+width_x0,y_0+height_y0),(0,255,0),2)

#   for xp_1, yp_1 in zip(xpix, ypix):
#     x_0 = int(xp_1)
#     y_0 = int(yp_1)
#     cv2.rectangle(img_read,(x_0-int(bounding_box),y_0-int(bounding_box)),(x_0+int(bounding_box),y_0+int(bounding_box)),(0,255,0),1)
    # cv2.rectangle(img_read,(int(xp_1-bounding_box),int(yp_1-bounding_box)),(int(xp_1+bounding_box),int(yp_1+bounding_box)),(0,255,0),1)

#   cv2.imwrite(r'data/custom/true_bb_images/'+str(i+1)+'.jpg',img_read)

#   cv2_imshow(img_read)


  with open("data/custom/labels/"+str(i+1)+".txt",'w+') as img_labels:
      
    img_read = cv2.imread(r'data/custom/images/'+str(i+1)+'.jpg')
    img_read = np.array(img_read)
    if "points" in class_names:
      for xp, yp in zip(xpix, ypix):
        img_labels.write("0"+" "+str(xp/width)+" "+str(yp/height)+" "+str(bounding_box/width)+" "+str(bounding_box/height)+"\n")
        x_0 = int(xp)
        y_0 = int(yp)
        cv2.rectangle(img_read,(x_0-int(bounding_box),y_0-int(bounding_box)),(x_0+int(bounding_box),y_0+int(bounding_box)),(0,255,0),1)
    
    if "ticks" in class_names:
      for x_j, y_j in x_tick_pos:
        img_labels.write("1"+" "+str(x_j/width)+" "+str(y_j/height)+" "+str(tick_box_size/width)+" "+str(tick_box_size/height)+"\n")

      for x_j, y_j in y_tick_pos:
        img_labels.write("1"+" "+str(x_j/width)+" "+str(y_j/height)+" "+str(tick_box_size/width)+" "+str(tick_box_size/height)+"\n")

    if "labels" in class_names:
      for item in x_label_coords:
        centre_x_label = item[0]
        centre_y_label = item[1]
        box_width_label = item[2]
        box_height_label = item[3]
        img_labels.write("2"+" "+str(centre_x_label/width)+" "+str(centre_y_label/height)+" "+str(box_width_label/width)+" "+str(box_height_label/height)+"\n")
        cv2.rectangle(img_read,(int(centre_x_label-box_width_label),int(centre_y_label-box_height_label)),(int(centre_x_label+box_width_label),int(centre_y_label+box_height_label)),(0,255,0),2)

      for item in y_label_coords:
        centre_x_label = item[0]
        centre_y_label = item[1]
        box_width_label = item[2]
        box_height_label = item[3]
        img_labels.write("2"+" "+str(centre_x_label/width)+" "+str(centre_y_label/height)+" "+str(box_width_label/width)+" "+str(box_height_label/height)+"\n")
        cv2.rectangle(img_read,(int(centre_x_label-box_width_label),int(centre_y_label-box_height_label)),(int(centre_x_label+box_width_label),int(centre_y_label+box_height_label)),(0,255,0),2)    

    cv2.imwrite(r'data/custom/true_bb_images/'+str(i+1)+'.jpg',img_read)
  plt.close(fig)



num_train = round(len(dataset)*train_test_ratio)
if "points" in class_names:
  print("points")
if "ticks" in class_names:
  print("ticks")
if "labels" in class_names:
  print("labels")
for i,d in enumerate(dataset):
  if i%500 == 0:
    print("Generated "+str(i+1))
  gen_scatterplot(d,x_dist_type[i],y_dist_type[i],i,x_distribution_param,y_distribution_param,total_samples,num_train)
  # break
