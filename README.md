# Frame-by-frame-reconstruction
Codes made for the video exhibition *Frame by frame reconstruction* at Galleri Blunk, Trondheim, 30 October – 1 November 2026.

## ---1: Lagrange interpolation----
*Husker du når gitaristen bare tok av? (Do you remember when the guitarist went crazy?)* — `lagrange.py`

To conjure up more frames than we have, we can use *Lagrange interpolation*. It creates a polynomial based on given points.

Our points are each R (red), G (green) and B (blue) value in each pixel at a position width w, height h. We have one frame like this for every time t, so we need to find polynomials for every pixel(w, h) that give the correct colour value at each time t.

For each time t we create a product over all the other times j ≠ t:

```
               n    (x - x_j)
  ℓ_t(x)  =    ∏   -----------
             j = 0  (x_t - x_j)
             j ≠ t
```

ℓ_t(x) says how much weight frame t gets at a new time x. So now we can use ℓ_t(x) to make a polynomial for each colour: p_n^R, p_n^G, p_n^B. We multiply ℓ_t(x) by the value of the colour c at that time: y_t, for every time t.

```
                n
  p_n^(c)(x) =  Σ   ℓ_t(x) · y_t
              t = 0
```

Now we can create a totally new frame for any time x we didn’t have originally.

## ---2: Polynomial regression with SVD----
*Hvordan reagerte han? (How did he react?)* — `regression_svd.py`

*Regression* creates all new frames and looks for the best approximating polynomial.

We can make a matrix b ∈ ℝ^(time × w·h·3) where the rows are time t (one row per frame) and the columns are the colour values R (red), G (green) and B (blue), repeating for each pixel at a position width w, height h.

```
             pixel(1, 1)  pixel(1, 2)
              R  G  B      R  G  B    . . .
           ┌──────────────────────────────┐
     t = 0 │         ---frame 0---        │
b =  t = 1 │         ---frame 1---        │
. . .
           └──────────────────────────────┘
```

So for every pixel(w, h), at every colour R, G, B, we want to create a polynomial p(t) = x_1 + x_2·t + x_3·t² + ... (here we decide the degree).

A ∈ ℝ^(time × (degree + 1)) is the Vandermonde matrix and only depends on time: row i is (1, t_i, t_i², ..., t_i^degree), while x ∈ ℝ^((degree + 1) × w·h·3) contains the coefficients of the polynomial.

Ax is the matrix representation of p(t). We have Ax ≈ b, and once we have found x, we can approximate a frame for any t. We find x = A⁺b, where A⁺ is the *Moore–Penrose inverse*, computed with *SVD (Singular Value Decomposition)*.

## ---3: Tensor CPD----
*Jeg savner Tinka (I miss Tinka)* & *Jeg vil huske pelsen (I want to remember the fur)* — `cp_tensor_decomposition.py`

A video can be made into a *tensor* X ∈ ℝ^(time × space × channel) where the space is width w times height h, and the channel has three values R (red), G (green) and B (blue). (R, G, B) becomes what we call a fiber for a pixel(t, w, h).

*Canonical polyadic decomposition* approximates X as a sum of rank 1 tensors:

```
        rank
  X  ≈   Σ   a^(s) ∘ b^(s) ∘ c^(s)   (∘ is the outer product)
       s = 1
```

We choose the rank according to how exact we want it. We have the matrices A ∈ ℝ^(time × rank), B ∈ ℝ^(space × rank), C ∈ ℝ^(channel × rank)

```
    a1 a2...          b1 b2...     c1 c2...
t=0 ┌       ┐ pixel 1 ┌       ┐ R ┌       ┐
t=1 │   A   │ pixel 2 │   B   │ G │   C   │
 ⋮  └       ┘    ⋮    └       ┘ B └       ┘
```

We find the values in A, B and C with the *ALS algorithm*, which starts with random values and repeats itself until the error converges.

The b^(s) ∘ c^(s), s = 1 ... rank, serve as “basis images”. Row t of A gives them different weights: frame(t) ≈ A(t, 1)·basis image 1 + ... + A(t, rank)·basis image rank. What is only there for a short moment gets washed out.

## ---4: Unit vectors and sorting algorithm----
*Skrudde jeg av strykejernet? (Did I turn off the iron?)* — `colour_degree_sorting.py`

For a pixel at width w and height h at a time t, pixel(t, w, h), we have three colour values R (red), G (green), B (blue) and want to find its hue (we ignore the brightness).

The three colours are placed as unit vectors on the hue colour wheel: red at 0°, green at 120° and blue at 240°:

```
  u_R = (cos 0°,   sin 0°)  = (1, 0)
  u_G = (cos 120°, sin 120°)
  u_B = (cos 240°, sin 240°)
```

We calculate v(t, w, h) = R·u_R + G·u_G + B·u_B\
and get hue = atan2(v_y, v_x) · 180/π  mod 360

For every position (w, h), we take the colours of pixel(w, h) at the different times t and sort them by hue with the *counting-sort algorithm*. We do this in windows of frames.

```
COUNTING-SORT(A, n, k)
 1  new: B[1 : n], C[0 : k]
 2  for i = 0 to k
 3      C[i] = 0
 4  for j = 1 to n
 5      C[A[j]] = C[A[j]] + 1
 6  for i = 1 to k
 7      C[i] = C[i] + C[i - 1]
 8  for j = n downto 1
 9      B[C[A[j]]] = A[j]
10      C[A[j]] = C[A[j]] - 1
11  return B
```

                      
