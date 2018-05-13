#PBS -l walltime=24:00:00 -N cvml
cd $PBS_O_WORKDIR

if [ -z "$1" ]
  then
    echo "No script supplied: qsub launcher.sh -F 'myscript.py'"
    exit 1
fi

echo Starting $1
source activate cvml
python $1
echo Script $1 ended

