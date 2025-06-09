#!/bin/bash
DT=0.1
NUMP_VALUES=(1024)
ACT_VALUES=(0 0.1 0.2)
DR_VALUES=(0.5 1 5)
SEED1_VALUES=(101)  
BOX_VALUES=(40)

OUTPUT_FILE="md_cell_vedio.py"

for i in "${!NUMP_VALUES[@]}"; do
    NUMP=${NUMP_VALUES[$i]}
    BOX=${BOX_VALUES[$i]}

    for ACT in "${ACT_VALUES[@]}"; do
        for DR in "${DR_VALUES[@]}"; do
            for SEED1 in "${SEED1_VALUES[@]}"; do
                JOB_SCRIPT="submit_${NUMP}_${ACT}_${DR}_${SEED1}_MIPSPYv.sh"
                echo "#!/bin/bash
#SBATCH --qos=serial
#SBATCH --exclude=node10
python ${OUTPUT_FILE} ${ACT} ${DR} ${DT} ${SEED1} ${NUMP} ${BOX}" > $JOB_SCRIPT
                chmod +x $JOB_SCRIPT
                sbatch $JOB_SCRIPT
                echo "已提交任务: ${OUTPUT_FILE} ${ACT} ${DR} ${DT} ${SEED1} ${NUMP} ${BOX}"
            done
        done
    done
done

echo "所有任务已提交！"

