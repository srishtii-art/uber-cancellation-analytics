SELECT COUNT(*) AS total_bookings
FROM rides;
SELECT
    booking_status,
    COUNT(*) AS total_bookings
FROM rides
GROUP BY booking_status
ORDER BY total_bookings DESC;