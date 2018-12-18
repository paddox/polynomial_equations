# polynomial_equations
Решение системы двух уравнений от двух переменных в рациональных числах.

В файле Polynomial_equations.py решается система двух уравнений в рациональных числах. В файле parallel.py сделана попытка распараллелить алгоритм, реализованный в первом файле.

На входе каждой программы две строки, записанные в файле input.txt, которые содержат в себе через запятую коэффициенты первого и второго уравнения соответственно. Уравнение от двух переменных имеет вид:

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Csum_%7Bi%3D0%2C%20j%3D0%7D%5E%7Bn%7D%20a_%7Bij%7Dx%5E%7Bi%7Dy%5E%7Bj%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\sum_{i=0, j=0}^{n} a_{ij}x^{i}y^{j}" width="104" height="53" />

Тогда в строку соответствующего уравнения в файле input.txt необходимо записать коэффициенты в следующем порядке:

<img src="http://www.sciweavers.org/tex2img.php?eq=a_%7Bn%2Cn%7D%2C%20a_%7Bn%2C%20n-1%7D%2C%20a_%7Bn%2C%20n-2%7D%2C%20...%2C%20a_%7Bn%2C%201%7D%2C%20a_%7Bn%2C%200%7D%2C%20a_%7Bn-1%2C%20n%7D%2C%20a_%7Bn-1%2C%20n-1%7D%2C%20...%2C%20a_%7B0%2C%200%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="a_{n,n}, a_{n, n-1}, a_{n, n-2}, ..., a_{n, 1}, a_{n, 0}, a_{n-1, n}, a_{n-1, n-1}, ..., a_{0, 0}" width="458" height="17" />

Дробные коэффициенты записываются через слэш, например, 1/2, -1/3
