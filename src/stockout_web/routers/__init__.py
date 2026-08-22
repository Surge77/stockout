"""HTTP handlers, one module per audience.

`auth` is for anybody, `forecast` is for a signed-in user, and `admin` is for an admin.
The split is by *who may call it* rather than by what it touches, because that is the
question a reviewer asks first and the one a misplaced route gets wrong.
"""
