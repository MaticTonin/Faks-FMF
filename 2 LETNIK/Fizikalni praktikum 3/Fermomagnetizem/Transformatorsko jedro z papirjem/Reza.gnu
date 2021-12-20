set terminal pdf
set out "Prvareza.pdf"
set title "Histereza magnetnega kroga z režo"
set ylabel "B[T]"
set xlabel "U_m[A]"
set key
m = 1000
n = 1/(16e-4*46)
stats "papir1.txt" u 4
min = STATS_min
max = STATS_max
f(x) = abs(x)
stats "<(sed -n '200,400p'  papir1.txt)" u (abs($2))

stats "<(sed -n '501,501p'  papir1.txt)" u (($4-(max+min)/2)*n)

plot 	"<(sed -n '251,251p'  papir1.txt)" u ($2*m):(($4-(max+min)/2)*n) \
	, "papir1.txt" u ($2*m):(($4-(max+min)/2)*n) w l