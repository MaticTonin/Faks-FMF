set terminal pdf
set out "Histereza_poskus.pdf"
set title "Histereza magnetnega kroga transformatorskega jekla in zeleza"
set ylabel "B[T]"
set xlabel "U_m[A]"
unset key
m = 1000
n = 1/(16e-4*46)
stats 'trans.txt' u 4
min = STATS_min
max = STATS_max
plot 'trans.txt' u ($2*m):(($4-(max+min)/2)*n) w l 