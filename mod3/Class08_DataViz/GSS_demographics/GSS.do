#delimit ;

   infix
      year     1 - 20
      id_      21 - 40
      hrs1     41 - 60
      hrs2     61 - 80
      marital  81 - 100
      childs   101 - 120
      income   121 - 140
      happy    141 - 160
      ballot   161 - 180
using GSS.dat;

label variable year     "Gss year for this respondent                       ";
label variable id_      "Respondent id number";
label variable hrs1     "Number of hours worked last week";
label variable hrs2     "Number of hours usually work a week";
label variable marital  "Marital status";
label variable childs   "Number of children";
label variable income   "Total family income";
label variable happy    "General happiness";
label variable ballot   "Ballot used for interview";


label define gsp001x
   99       "No answer"
   98       "Don't know"
   -1       "Not applicable"
;
label define gsp002x
   99       "No answer"
   98       "Don't know"
   -1       "Not applicable"
;
label define gsp003x
   9        "No answer"
   5        "Never married"
   4        "Separated"
   3        "Divorced"
   2        "Widowed"
   1        "Married"
;
label define gsp004x
   9        "Dk na"
   8        "Eight or more"
;
label define gsp005x
   99       "No answer"
   98       "Don't know"
   13       "Refused"
   12       "$25000 or more"
   11       "$20000 - 24999"
   10       "$15000 - 19999"
   9        "$10000 - 14999"
   8        "$8000 to 9999"
   7        "$7000 to 7999"
   6        "$6000 to 6999"
   5        "$5000 to 5999"
   4        "$4000 to 4999"
   3        "$3000 to 3999"
   2        "$1000 to 2999"
   1        "Lt $1000"
   0        "Not applicable"
;
label define gsp006x
   9        "No answer"
   8        "Don't know"
   3        "Not too happy"
   2        "Pretty happy"
   1        "Very happy"
   0        "Not applicable"
;
label define gsp007x
   4        "Ballot d"
   3        "Ballot c"
   2        "Ballot b"
   1        "Ballot a"
   0        "Not applicable"
;


label values hrs1     gsp001x;
label values hrs2     gsp002x;
label values marital  gsp003x;
label values childs   gsp004x;
label values income   gsp005x;
label values happy    gsp006x;
label values ballot   gsp007x;


