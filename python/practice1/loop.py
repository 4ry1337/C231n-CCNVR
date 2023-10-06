""" animals = ["cat", "dog", "monkey"]
for animal in animals:
    print(animal) """


""" animals = ["cat", "dog", "monkey"]
en = enumerate(animals)
print(list(en))
for idx, animal in enumerate(animals):
    print("#%d: %s" % (idx + 1, animal)) """


""" nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)   # Prints [0, 1, 4, 9, 16] """

nums = [0, 1, 2, 3, 4]
squares = [x**2 for x in nums if x % 2 == 0]
print(squares)  # Prints [0, 1, 4, 9, 16]
