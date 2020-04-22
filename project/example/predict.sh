# Create a backup of the original Dockerfile
if [[ -f ../prediction/Dockerfile.BAK ]]
then
    echo "Backup already done. Going on"
else
    mv ../prediction/Dockerfile ../prediction/Dockerfile.BAK
fi
# Replace the new Dockerfile with the original one
cp prediction ../prediction/Dockerfile
# Build the docker-compose images and run
cd .. && sudo docker-compose up --build influx prediction
# Restore the original Dockerfile
cd example
./restore_original.sh