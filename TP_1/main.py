import cv2 as cv
from sift import sift, octave_gaussian, DOG_pyramid

image = cv.imread('./images/ex_1.jpg', cv.IMREAD_COLOR)

# sift(image=image)

pyramid_gauss = octave_gaussian(image)

print(len(pyramid_gauss['level1']))

i = 0
for key, imgs in pyramid_gauss.items():
    for x in imgs:
        # cv.imshow(f"Image {i}", x)
        i += 1

pyramid_dog = DOG_pyramid(pyramid_gauss, 4)
# for i, imgs in pyramid_dog.items():
#     for x in imgs:
#         cv.imshow(f'DOG Pyramid Image {i}', x)


for i, imgs in pyramid_dog.items():
    for x in imgs:
        cv.imshow(f'Image with key points {i}', sift(x))






cv.waitKey(0)
cv.destroyAllWindows()






