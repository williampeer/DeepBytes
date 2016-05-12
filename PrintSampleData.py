import Tools
from PIL import Image
from DataWrapper import training_patterns_heterogeneous


images = []
for i in range(10):
    cur_im = Tools.create_image_helper(training_patterns_heterogeneous[i][1])
    images.append(cur_im)

aggregate_im = Image.new('1', (7 * 8 * 10, 7 * 8))
for im_ctr in range(len(images)):
    aggregate_im.paste(images[im_ctr], (7 * 8 * im_ctr + im_ctr, 0))

aggregate_im.show()

im_2 = Image.new('1', (7*8*10+2, 7*8*2+4))
im_2.paste(aggregate_im, (1, 1))
im_2.paste(aggregate_im, (1, 7*8+2))
# im_2.show()

aggregate_im.save('aggregate_im_lowercase.png', 'PNG')
# im_2.save('im_both2.png', 'PNG')
