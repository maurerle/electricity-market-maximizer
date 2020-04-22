# Create a backup of the original Dockerfile
if [[ -f ../spiders/Dockerfile.BAK ]]
then
    echo "Backup already done. Going on"
else
    mv ../spiders/Dockerfile ../spiders/Dockerfile.BAK
fi
# Replace the new Dockerfile with the original one
cp createdb ../spiders/Dockerfile
# Build the docker-compose images and run
cd .. && sudo docker-compose up --build influx spiders
# Restore the original Dockerfile
cd example
./restore_original.sh