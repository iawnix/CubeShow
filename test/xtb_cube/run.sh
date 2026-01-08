

A_TO_BOHR=1.8897259886

function create_density_xtb_inp(){
	cat <<- EOF > density.inp
	\$cube
	  step=$(echo "scale=10; $A_TO_BOHR * 0.2" | bc)
	  boff=6
	  pthr=0.0
	\$end
	\$write
	  density=true
	  spin density=false
	  fod=false
	  charges=false
	  mulliken=false
	  geosum=false
	  inertia=false
	  mos=false
	  wiberg=false
	\$end
EOF
}

function cal_density_xtb(){
	local chrg=$1
	local uhf=$2
	local sdf=$3

	xtb $sdf \
		--chrg $chrg --uhf $uhf \
		--norestart \
		--input density.inp \
		--gfn 2 \
		> density.log 2>&1
}


#+++++++++++++++++++++# TEST #+++++++++++++++++++++++#
test_chrg=0
test_uhf=0
test_sdf='xtbopt.sdf'

create_density_xtb_inp
cal_density_xtb $test_chrg $test_uhf $test_sdf
#++++++++++++++++++++++++++++++++++++++++++++++++++++#

