import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.

input_path = "/media/cvpr/CM_22/KR_muliple/"
output_pth = "/media/cvpr/CM_22/tiny_KR"

splitfolders.ratio(input_path, output=output_pth, seed=1337, ratio=(.3,.3,.4), group_prefix=None, move=True) # default values
