from django.shortcuts import render
from django.core.mail import send_mail


# Create your views here.
def contacts(request):
    if request.method == "POST":
        fname = request.POST['fname']
        mail = request.POST['mail']
        recip = request.POST['recip']
        obj = request.POST['obj']
        message = "The following message has been sent from : %s \n\n" %mail + "%s" %request.POST['subject'] # this is a good way to use the code already written to send email from the EMAIL_HOST defined in settings. 
        message = message + "\n\nRemember to answer to %s" %fname + " at %s" %mail
        mailTo= []
        mailTo.append(recip)
        print(mailTo)
        print(message )
        

        # send an email 
        send_mail(
            obj,   # what is usually called OGGETTO / molti lo chiamo subject
            message , #message - The real message of the mail with our processing to understand who is writing
            mail, # EMAIL_HOST_USER in settings.py
            mailTo,  #recipient LIST 
        )
        return render(request, 'contacts/page.html', {'fname' : fname})
    else:
        return render(request, 'contacts/page.html', {})
