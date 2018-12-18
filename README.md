# polynomial_equations
Решение системы двух уравнений от двух переменных в рациональных числах.

В файле Polynomial_equations.py решается система двух уравнений в рациональных числах. В файле parallel.py сделана попытка распараллелить алгоритм, реализованный в первом файле.

На входе каждой программы две строки, записанные в файле input.txt, которые содержат в себе через запятую коэффициенты первого и второго уравнения соответственно. Уравнение от двух переменных имеет вид:

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i=0,&space;j=0}^{n}&space;a_{ij}x^{i}y^{j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i=0,&space;j=0}^{n}&space;a_{ij}x^{i}y^{j}" title="\sum_{i=0, j=0}^{n} a_{ij}x^{i}y^{j}" /></a>

Тогда в строку соответствующего уравнения в файле input.txt необходимо записать коэффициенты в следующем порядке:

<a href="https://www.codecogs.com/eqnedit.php?latex=a_{n,n},&space;a_{n,&space;n-1},&space;a_{n,&space;n-2},&space;...,&space;a_{n,&space;1},&space;a_{n,&space;0},&space;a_{n-1,&space;n},&space;a_{n-1,&space;n-1},&space;...,&space;a_{0,&space;0}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_{n,n},&space;a_{n,&space;n-1},&space;a_{n,&space;n-2},&space;...,&space;a_{n,&space;1},&space;a_{n,&space;0},&space;a_{n-1,&space;n},&space;a_{n-1,&space;n-1},&space;...,&space;a_{0,&space;0}" title="a_{n,n}, a_{n, n-1}, a_{n, n-2}, ..., a_{n, 1}, a_{n, 0}, a_{n-1, n}, a_{n-1, n-1}, ..., a_{0, 0}" /></a>

Дробные коэффициенты записываются через слэш, например, 1/2, -1/3
