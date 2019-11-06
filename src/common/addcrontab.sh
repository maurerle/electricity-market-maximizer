cp -r ../smartgrids ~
cd ~/smartgrids
crontab -l > crontab
echo 'PATH=/usr/local/bin/:/usr/bin:/usr/sbin' >> crontab
echo '0 17 * * * cd ~/smartgrids && make getDaily >> ~/smartgrids/crontab.log 2>&1' >> crontab
crontab crontab
rm crontab
echo 'Daily cronjob scheduled at 17:00'