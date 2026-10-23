# Frame-by-frame-reconstruction
Codes made for a video exhibition at galleri blunk:  Frame by frame reconstruction

## ---1: Lagrange interpolation----

To conjure up more frames than we have, we can use Lagrange interpolation. It creates a polynomial based on given points.

Our points are each R (red), G (green) and B (blue) value in each pixel at a position width w, height h. We have one frame like this for every time t, so we are looking for a polynomial that gives the correct colour value at each time t.

For each time t we create
```
               n    (x - x_j)
  ℓ_t(x)  =    ∏   -----------
             j = 0  (x_t - x_j)
             j ≠ t
```

So now we can use this for every pixel(w,h), to make a polynomial for each colour going through that position, p_n^R, p_n^G, p_n^B, using
```
                n
  p_n^(c)(x) =  Σ   ℓ_t(x) · y_t
              t = 0
```

where y_t is the value of the colour c at that time.

Now we can create a totally new frame for any new time x.

## ---2: Polynomial regression with SVD----

Regression creates all new frames and looks for the best approximating polynomial.

We can make a matrix where the rows are time t and the columns are the colour values R (red), G (green) and B (blue), repeating for each pixel at a position width w, height h.

```
             pixel(1,1)   pixel(1,2)
              R  G  B      R  G  B    . . .
           ┌──────────────────────────────┐
     t = 0 │         ---frame 1---        │
b =  t = 1 │         ---frame 2---        │
. . .
           └──────────────────────────────┘
```

b ∈ ℝ^(number of frames × w·h·3)

So for every pixel (w,h), at every colour R, G, B, we create a polynomial p(t) = x_1 + x_2·t + x_3·t² + ... (here we decide the degree).

A ∈ ℝ^(number of frames × (degree + 1)) is the Vandermonde matrix and only depends on time: row i is (1, t_i, t_i², …, t_i^degree), while x ∈ ℝ^((degree + 1) × w·h·3) contains the coefficients of the polynomial.

Ax is the matrix representation of p(t). We have Ax ≈ b, and once we have found x, we can approximate a frame for any t. We find x = A⁺b, where A⁺ is the Moore–Penrose inverse (pseudoinverse), computed with SVD (Singular Value Decomposition).

## ---3: Tensor CPD----


A video can be made into a tensor X ∈ ℝ^(time × space × channel) where the space is width w times height h, and the channel has three values R (red), G (green) and B (blue). (R, G, B) becomes what we call a fiber for a pixel_(t,w,h).

Canonical polyadic decomposition approximates X as a sum of rank 1 tensors:

```
  rank
X ≈ Σ  a^(s) ⊚ b^(s) ⊚ c^(s)   (⊚ is the outer product)
   s=1
```

We choose the rank according to how exact we want it. We have the matrices A ∈ ℝ^(time × rank), B ∈ ℝ^(space × rank), C ∈ ℝ^(channel × rank)

```
   a1 a2 …             b1 b2 …      c1 c2 …
t=0 ┌       ┐ pixel 1 ┌       ┐ R ┌       ┐
t=1 │   A   │ pixel 2 │   B   │ G │   C   │
 ⋮  └       ┘    ⋮    └       ┘ B └       ┘
```

We find the values in A, B and C with the ALS algorithm, which starts with random values and repeats itself until the error converges.

For every s = 1 ... rank, b^(s) ∘ c^(s) becomes a base image. Each frame is now made of its row in A as weights for each base image: frame(t) ≈ A(t,1) · base image 1 + ... + A(t,rank) · base image rank. What is only there for a short moment is averaged away.

## ---4: Unit vectors and sorting algorithm----

For a pixel with width w and height h at a time t, X(t,w,h), we have three colour values R (red), G (green), B (blue) and want to find its hue (we ignore the brightness).

The three colours are placed as unit vectors on the hue colour wheel: red at 0°, green at 120° and blue at 240°:
```
  u_R = (cos 0°,   sin 0°)  = (1, 0)
  u_G = (cos 120°, sin 120°)
  u_B = (cos 240°, sin 240°)
```

We calculate h(t,w,h) = R·u_R + G·u_G + B·u_B\
and get hue = atan2(h_y, h_x) · 180/π  mod 360

Now we want to sort, for every position w,h, every pixel for every t, using the counting-sort algorithm. We do this in intervals.

```
COUNTING-SORT(A, n, k)
 1  let B[1:n] and C[0:k] be new arrays
 2  for i = 0 to k
 3      C[i] = 0
 4  for j = 1 to n
 5      C[A[j]] = C[A[j]] + 1
 6  // C[i] now contains the number of elements equal to i
 7  for i = 1 to k
 8      C[i] = C[i] + C[i-1]
 9  // C[i] now contains the number of elements
10  // less than or equal to i
11  for j = n downto 1
12      B[C[A[j]]] = A[j]
13      C[A[j]] = C[A[j]] - 1
14  return B
```





                      
