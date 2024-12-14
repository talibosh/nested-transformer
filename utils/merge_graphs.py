from PIL import Image, ImageDraw, ImageFont
import os


def merge_images_with_labels(image_list, output_path):
    # Open images
    images = [Image.open(img) for img in image_list]

    # Ensure there are 6 images
    #if len(images) != 6:
    #    raise ValueError("Exactly 6 images are required.")
    if len(images) != 3:
        raise ValueError("Exactly 6 images are required.")

    # Get the width and height of each image (assuming all are the same size)
    width, height = images[0].size

    # Create a blank image for the grid (2 columns, 3 rows, plus space for text labels)
    grid_width = 2 * width + 100  # Extra space for text on the left side
    grid_height = 3 * height + 100  # Extra space for labels below
    grid_width = 1 * width  # Extra space for text on the left side
    grid_height = 3 * height  # Extra space for labels below

    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))  # White background

    # Paste images into the grid
    for i, img in enumerate(images):
        # Calculate the position for each image in the 2x3 grid
        #x = 100 + (i % 2) * width  # Shift to the right to leave space for labels on the left
        #y = (i // 2) * height

        x = (i % 3) * width  # Shift to the right to leave space for labels on the left
        x = 0
        y = (i % 3) * height
        grid_image.paste(img, (x, y))

    # Create a drawing object to add text
    draw = ImageDraw.Draw(grid_image)

    # Optional: Load a font (Pillow's default font if you don't have a specific one)
    '''
    try:
        font = ImageFont.truetype("arial.ttf", 40)  # Try to load a custom font
    except IOError:
        font = ImageFont.load_default()  # Default font if custom not found

    # Add text on the left side, next to each row (centered vertically in the row)
    text_labels = ["Cats",  "Horses", "Dogs"]
    for i, label in enumerate(text_labels):
        text_y = i * height + height // 2 - 20  # Vertically centered in the row
        draw.text((20, text_y), label, font=font, fill="black")  # Place on the left side

    # Add 'A' and 'B' below the grid (centered in columns)
    draw.text((100 + width // 2 - 10, 3 * height + 30), "A", font=font, fill="black")  # Centered below column 1
    draw.text((100 + 3 * width // 2 - 10, 3 * height + 30), "B", font=font, fill="black")  # Centered below column 2
    '''
    # Save the final grid image
    grid_image.save(output_path)



def merge_images_hm(image_list, output_path):
    # Open images
    images1 = [Image.open(img) for img in image_list]

    # Ensure there are 6 images
    #if len(images) != 6:
    #    raise ValueError("Exactly 6 images are required.")
    if len(images1) != 30:
        raise ValueError("Exactly 30 images are required.")

    images = [img.resize((224, 224)) for img in images1]

    # Get the width and height of each image (assuming all are the same size)
    width, height = images[0].size

    # Create a blank image for the grid (2 columns, 3 rows, plus space for text labels)
    grid_width = 5 * width   # Extra space for text on the left side
    grid_height = 6 * height   # Extra space for labels below

    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))  # White background

    # Paste images into the grid
    for i, img in enumerate(images):
        # Calculate the position for each image in the 2x3 grid
        #x = 100 + (i % 2) * width  # Shift to the right to leave space for labels on the left
        #y = (i // 2) * height

        x = (i % 5) * width  # Shift to the right to leave space for labels on the left
        y = (i // 5) * height
        grid_image.paste(img, (x, y))

    # Create a drawing object to add text
    draw = ImageDraw.Draw(grid_image)

    # Optional: Load a font (Pillow's default font if you don't have a specific one)

    # Save the final grid image
    grid_image.save(output_path)



import matplotlib.pyplot as plt
import numpy as np
def plot_graph_4_ijcv( y_values):

# X-axis categories
    categories = ['Ears', 'Eyes', 'Mouth']
    x_positions = np.arange(len(categories))

    # Data points (y-axis values)
    #y_values = np.random.randint(0, 4, 9)

    # Colors for different categories
    colors = ['red', 'green', 'blue']
    animals = ['Cats', 'Horses', 'Dogs']

    # Shapes for different parts
    shapes = ['^', 'o', '*']
    parts = ['Ears', 'Eyes', 'Mouth']

    # Create the plot
    fig, ax = plt.subplots()

    # Plotting the points
    for i, color in enumerate(colors):
        for j, shape in enumerate(shapes):
            ax.scatter(x_positions[j], y_values[j*3+i], color=color, marker=shape, s=100)

    # Set the x-axis ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories)

    # Set y-axis limits
    ax.set_ylim(0, 5)
    ax.set_xlabel('segment')
    ax.set_ylabel('segs quality score')

    # Adding legends
    #legend_shapes = [plt.Line2D([0], [0], marker=s, color='w', markerfacecolor='black', markersize=10) for s in shapes]
    #legend_colors = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10) for c in colors]

    # Legend for shapes
    #shape_legend = ax.legend(legend_shapes, parts, title="Shape", loc="upper right", bbox_to_anchor=(1.2, 1))
    # Legend for colors
    #color_legend = ax.legend(legend_colors, animals, title="DataSet", loc="upper right", bbox_to_anchor=(1.2, 0.8))

    # Add legends to plot
    #ax.add_artist(shape_legend)
    segs_name = {'top': 'Ears', 'middle': 'Eyes', 'bottom': 'Muzzles',
                     'ear': 'Ears','eye':'Eyes', 'ears':'Ears', 'eyes':'Eyes','mouth':'Mouth'}

    segments_marks={'eyes':'o','ears':'^','mouth':'*'}
    for i, seg in enumerate(segments_marks.keys()):
        ax.scatter([], [], edgecolors='k',facecolors='none', marker=segments_marks[seg], s=100, label=segs_name[seg])

    #if not(seg=='scaled_score'):
    #    ax.scatter([], [], edgecolors='k',facecolors='none', marker=segments_marks[seg], s=100, label=segs_name[seg])

    animal_colors={'Cats':'red', 'Horses':'green', 'Dogs':'blue'}
    animal_name={'Cats':'Cats', 'Horses':'Horses','Dogs':'Dogs'}
    for i, net in enumerate(animal_colors.keys()):
        ax.scatter([], [], color=animal_colors[net], marker='s', s=100, label=animal_name[net])


    ax.legend(loc='upper right')


    # Show the plot
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('/home/tali/segs_all_sets.jpg')
    plt.show()
# Example usage:
main_dir = '/home/tali'
cats_dir = os.path.join(main_dir,'cats_pain_proj')
horses_dir = os.path.join(main_dir,'horses')
dogs_dir = os.path.join(main_dir,'dogs_annika_proj')
dirs = [cats_dir, horses_dir, dogs_dir]
image_list = []

#cats
d = os.path.join(cats_dir,'face_images','masked_images')
image_list.append(os.path.join(d, 'pain', 'cat_12_video_3.9.jpg'))
d_maps = os.path.join(cats_dir, 'pytorch_dino', 'maps')
image_list.append(os.path.join(d_maps, 'grad_cam','12', 'Yes', 'cat_12_video_3.9_.jpg'))
image_list.append(os.path.join(d_maps, 'xgrad_cam','12', 'Yes', 'cat_12_video_3.9_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','12', 'Yes', 'cat_12_video_3.9_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','12', 'Yes', 'cat_12_video_3.9_power.jpg'))

#img 2
image_list.append(os.path.join(d, 'no_pain', 'cat_13_video_1.10.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam','13', 'No', 'cat_13_video_1.10_.jpg'))
image_list.append(os.path.join(d_maps, 'xgrad_cam','13', 'No', 'cat_13_video_1.10_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','13', 'No', 'cat_13_video_1.10_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','13', 'No', 'cat_13_video_1.10_power.jpg'))

#horses
d = os.path.join(horses_dir,'dataset')
image_list.append(os.path.join(d, 'Yes','masked_images', '35-15C4.jpg'))
d_maps = os.path.join(horses_dir, 'pytorch_dino', 'maps')
image_list.append(os.path.join(d_maps, 'grad_cam','15', 'Yes', '35-15C4_.jpg'))
image_list.append(os.path.join(d_maps, 'xgrad_cam','15', 'Yes', '35-15C4_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','15', 'Yes', '35-15C4_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','15', 'Yes', '35-15C4_power.jpg'))

#img 2
image_list.append(os.path.join(d, 'No','masked_images', '73-29C1.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam','29', 'No', '73-29C1_.jpg'))
image_list.append(os.path.join(d_maps, 'xgrad_cam','29', 'No', '73-29C1_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','29', 'No', '73-29C1_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','29', 'No', '73-29C1_power.jpg'))

#dogs
d = os.path.join(dogs_dir,'data_set')
image_list.append(os.path.join(d, '29','N','337', 'masked_images', '00000060.jpg'))
d_maps = os.path.join(dogs_dir, 'pytorch_dino', 'maps')
image_list.append(os.path.join(d_maps, 'grad_cam','29', 'N','337', '00000060_.jpg'))
image_list.append(os.path.join(d_maps, 'xgrad_cam','29', 'N','337', '00000060_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','29', 'N','337', '00000060_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','29', 'N','337', '00000060_power.jpg'))

#img 2
image_list.append(os.path.join(d, '26','N', '70136223','masked_images', '00000077.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam','26', 'N', '70136223', '00000077_.jpg'))
image_list.append(os.path.join(d_maps, 'xgrad_cam','26', 'N', '70136223', '00000077_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','26', 'N', '70136223', '00000077_.jpg'))
image_list.append(os.path.join(d_maps, 'grad_cam_plusplus','26', 'N', '70136223', '00000077_power.jpg'))

output_path = '/home/tali/hms.jpg'
merge_images_hm(image_list, output_path)

y_values= [0.577, 0.924, 0.771, 2.511, 4.855, 2.08, 0.675, 0.348, 1.14]
#= [0.577, 0.924,0.771,2.511,  4.855, 2.08, 0.675,  0.348, 1.14]
plot_graph_4_ijcv( y_values )

reg_plots = "plots_new"
restrained_plots = "plots_restrain"
images_names_quals = ["quals_all.jpg", "seg_quals_all.jpg"]
images_names_scaled = ["scaled_all.jpg", "seg_scaled_all.jpg"]
image_list = []
for d in dirs:
    image_list.append(os.path.join(d,reg_plots, "quals_all.jpg"))
output_path = os.path.join(main_dir, "quals_3.jpg")
merge_images_with_labels(image_list, output_path)

for d in dirs:
    for n in images_names_quals:
        image_list.append(os.path.join(d,reg_plots, n))
output_path = os.path.join(main_dir, "quals_reg.jpg")
merge_images_with_labels(image_list, output_path)
image_list = []
for d in dirs:
    for n in images_names_scaled:
        image_list.append(os.path.join(d,reg_plots, n))
output_path = os.path.join(main_dir, "scaled_reg.jpg")
merge_images_with_labels(image_list, output_path)

image_list = []
for d in dirs:
    for n in images_names_quals:
        image_list.append(os.path.join(d,restrained_plots, n))
output_path = os.path.join(main_dir, "quals_restrained.jpg")
merge_images_with_labels(image_list, output_path)
image_list = []
for d in dirs:
    for n in images_names_scaled:
        image_list.append(os.path.join(d,restrained_plots, n))
output_path = os.path.join(main_dir, "scaled_restrained.jpg")
merge_images_with_labels(image_list, output_path)

