import Tools
from PIL import Image
from DataWrapper import training_patterns_associative, training_patterns_heterogeneous


images_ass = []
for i in range(25):
    cur_im = Tools.create_image_helper(training_patterns_associative[i][1])
    images_ass.append(cur_im)

aggregate_im_ass = Image.new('1', (7 * 8 * 25+25+2, 7 * 8))
for im_ctr in range(len(images_ass)):
    aggregate_im_ass.paste(images_ass[im_ctr], (7 * 8 * im_ctr + im_ctr, 0))

images_hetero = []
for i in range(25):
    cur_im = Tools.create_image_helper(training_patterns_heterogeneous[i][1])
    images_hetero.append(cur_im)

aggregate_im_hetero = Image.new('1', (7 * 8 * 25 + 2 + 25, 7 * 8))
for im_ctr in range(len(images_hetero)):
    aggregate_im_hetero.paste(images_hetero[im_ctr], (7 * 8 * im_ctr + im_ctr, 0))

# aggregate_im.show()

im_2 = Image.new('1', (7*8*25+25+2, 7*8*2+4))
im_2.paste(aggregate_im_ass, (1, 1))
im_2.paste(aggregate_im_ass, (1, 7 * 8 + 2))
# im_2.show()

im_3 = Image.new('1', (7*8*25+25+2, 7*8*2+4))
im_3.paste(aggregate_im_ass, (1, 1))
im_3.paste(aggregate_im_hetero, (1, 7 * 8 + 2))
# im_2.show()


aggregate_im_hetero.save('aggregate_im_lowercase.png', 'PNG')
aggregate_im_ass.save('aggregate_im_UPPERCASE.png', 'PNG')
im_2.save('im_both_ass.png', 'PNG')
im_3.save('im_both_hetero.png', 'PNG')
